"""Preprocess one new optical scene and append its tiles to prototype artifacts.

This deliberately accepts one acquisition at a time. It stores the image tiles
and manifest without rebuilding existing OSCD artifacts. If an embedding model,
vector store, and tile repository are supplied, it also appends searchable tile
embeddings and date metadata.
"""
from __future__ import annotations

from datetime import date as Date
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.tiling import tile_scene


def _load_scene(path: Path) -> np.ndarray:
    """Load RGB imagery from NPY/NPZ, common image formats, or GeoTIFF."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "image" if "image" in archive.files else archive.files[0]
            image = archive[key]
    elif suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError("GeoTIFF ingestion requires rasterio") from exc
        with rasterio.open(path) as src:
            if src.count < 3:
                raise ValueError("Scene must contain at least three RGB bands")
            image = np.moveaxis(src.read([1, 2, 3]), 0, -1)
    else:
        from PIL import Image
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected an HxWx3 scene, got shape {image.shape}")
    image = image[..., :3]
    if not np.isfinite(image).all():
        raise ValueError("Scene contains NaN or infinite pixel values")
    return image


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def ingest_single_scene(
    scene_path: Path,
    location_id: str,
    date: str,
    sensor: str = "sentinel-2",
    *,
    output_dir: Optional[Path] = None,
    tile_size: int = 256,
    bbox: Optional[Dict[str, float]] = None,
    embedding_model: Any = None,
    vector_store: Any = None,
    vector_index_path: Optional[Path] = None,
    tile_repo: Any = None,
    allow_same_tile_across_dates: bool = False,
) -> Dict[str, Any]:
    """Tile and persist one RGB acquisition; optionally append search metadata.

    When ``embedding_model`` and ``vector_store`` are provided, each tile is
    encoded and appended to the supplied in-memory index. ``vector_index_path``
    makes that append durable using a same-directory atomic replace. A tile
    repository can be supplied to persist the acquisition date and bbox.

    ``allow_same_tile_across_dates`` is for a multi-temporal sequence on a
    fixed spatial grid. It permits the same tile ID to be appended on a new
    date while still rejecting a duplicate tile/date pair. Vector IDs include
    the date in this mode so each acquisition remains independently searchable.
    """
    path = Path(scene_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scene file not found: {path}")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    try:
        acquisition_date = Date.fromisoformat(date).isoformat()
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD format") from exc
    location = re.sub(r"[^a-zA-Z0-9-]+", "", location_id.strip().lower())
    normalized_sensor = re.sub(r"[^a-zA-Z0-9-]+", "", sensor.strip().lower())
    if not location or not normalized_sensor:
        raise ValueError("location_id and sensor must contain letters or numbers")
    if (embedding_model is None) != (vector_store is None):
        raise ValueError("embedding_model and vector_store must be supplied together")
    if vector_index_path is not None and vector_store is None:
        raise ValueError("vector_index_path requires embedding_model and vector_store")
    if tile_repo is not None and bbox is None:
        raise ValueError("bbox is required when persisting tile metadata")

    output_root = (
        Path(output_dir)
        if output_dir
        else Path(os.getenv("TERRAEYES_DATA_DIR", "./data")) / "incremental_tiles"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = {"tiles": []}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest.get("tiles"), list):
            raise ValueError(f"Invalid incremental manifest: {manifest_path}")

    image = _load_scene(path)
    pipeline = PreprocessingPipeline(sensor=normalized_sensor)
    processed = pipeline.run(image)["image"]
    tiles = tile_scene(
        processed,
        tile_size=tile_size,
        location_id=location,
        sensor=normalized_sensor,
    )
    if not tiles:
        raise ValueError("Scene is too small to produce any tiles")

    tile_ids = [tile_id for tile_id, _, _ in tiles]
    vector_ids = (
        [f"{tile_id}@{acquisition_date}" for tile_id in tile_ids]
        if allow_same_tile_across_dates
        else tile_ids
    )
    if allow_same_tile_across_dates:
        known_dates = {
            (row.get("tile_id"), row.get("date")) for row in manifest["tiles"]
        }
        duplicate_ids = {
            tile_id for tile_id in tile_ids
            if (tile_id, acquisition_date) in known_dates
        }
    else:
        known = {row.get("tile_id") for row in manifest["tiles"]}
        duplicate_ids = known.intersection(tile_ids)
    indexed_ids = set(getattr(vector_store, "_ids", [])) if vector_store is not None else set()
    duplicate_ids.update(indexed_ids.intersection(vector_ids))
    if duplicate_ids:
        raise ValueError(f"Tile IDs already exist in the manifest or vector index: {sorted(duplicate_ids)}")

    vectors = None
    if embedding_model is not None:
        vectors = np.stack([
            embedding_model.encode_image(tile.astype(np.float32))
            for _, tile, _ in tiles
        ]).astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(tiles):
            raise ValueError("Embedding model returned an invalid vector batch")

    city_dir = output_root / location
    city_dir.mkdir(parents=True, exist_ok=True)
    new_rows = []
    written_paths = []
    try:
        for (tile_id, tile, bounds), vector_id in zip(tiles, vector_ids):
            tile_path = city_dir / f"{tile_id}_{acquisition_date}.npz"
            if tile_path.exists():
                raise FileExistsError(f"Tile artifact already exists: {tile_path}")
            np.savez_compressed(
                tile_path,
                image=tile.astype(np.float32),
                date=acquisition_date,
                sensor=normalized_sensor,
                source=str(path.resolve()),
            )
            written_paths.append(tile_path)
            row = {
                "tile_id": tile_id,
                "location_id": location,
                "date": acquisition_date,
                "sensor": normalized_sensor,
                "embedding_id": vector_id,
                "bounds_pixels": list(bounds),
                "shape": list(tile.shape),
                "array_path": tile_path.relative_to(output_root).as_posix(),
                "source": str(path.resolve()),
            }
            new_rows.append(row)

        if vector_store is not None:
            vector_store.add(vectors, vector_ids)

        if tile_repo is not None:
            from app.db.models import TileRecord
            for row in new_rows:
                tile_repo.insert_tile(TileRecord(
                    tile_id=row["tile_id"],
                    location_id=location,
                    sensor=normalized_sensor,
                    date=acquisition_date,
                    bbox_json=json.dumps(bbox),
                    embedding_id=row["embedding_id"],
                    ingested_at=f"{acquisition_date}T00:00:00Z",
                ))

        updated_manifest = {**manifest, "tiles": manifest["tiles"] + new_rows}
        _atomic_json(manifest_path, updated_manifest)

        if vector_index_path is not None:
            index_path = Path(vector_index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            temp_index = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp.npz")
            try:
                vector_store.save(str(temp_index))
                os.replace(temp_index, index_path if index_path.suffix == ".npz" else Path(f"{index_path}.npz"))
            finally:
                if temp_index.exists():
                    temp_index.unlink()
    except Exception:
        for tile_path in written_paths:
            tile_path.unlink(missing_ok=True)
        raise

    return {
        "status": "success",
        "location_id": location,
        "date": acquisition_date,
        "sensor": normalized_sensor,
        "tiles_created": len(tiles),
        "tile_ids": tile_ids,
        "vector_ids": vector_ids if embedding_model is not None else [],
        "manifest_path": str(manifest_path),
        "index_size": getattr(vector_store, "size", None),
    }
