"""Build the searchable DynamicEarthNet index from official training AOIs."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend", ROOT / "data-handling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters.dynamicearthnet.adapter import DynamicEarthNetAdapter
from app.db.connection import DatabaseManager
from app.db.models import TileRecord
from app.db.repositories.tile_repo import TileRepository
from app.models.embedding.clip_vit_b32 import CLIPViTB32EmbeddingModel
from app.vector_store.faiss_store import FaissVectorStore


def _training_area_ids(splits_csv: Path) -> set[str]:
    ids = set()
    with splits_csv.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("split") != "train":
                continue
            parts = row.get("planet_path", "").replace("\\", "/").rstrip("/").split("/")
            if len(parts) >= 2 and parts[-2]:
                ids.add(parts[-2])
    return ids


def _tile_bbox(adapter: DynamicEarthNetAdapter, location: str, row: int, col: int) -> tuple[dict[str, float], str]:
    crs, transform = adapter.get_area_georeferencing(location)
    x0, px, rx, y0, ry, py = transform
    left, top = col * 256, row * 256
    corners = [
        (x0 + left * px + top * rx, y0 + left * ry + top * py),
        (x0 + (left + 256) * px + top * rx, y0 + (left + 256) * ry + top * py),
        (x0 + left * px + (top + 256) * rx, y0 + left * ry + (top + 256) * py),
        (x0 + (left + 256) * px + (top + 256) * rx, y0 + (left + 256) * ry + (top + 256) * py),
    ]
    project = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geographic = [project.transform(x, y) for x, y in corners]
    bbox = {
        "west": min(point[0] for point in geographic),
        "south": min(point[1] for point in geographic),
        "east": max(point[0] for point in geographic),
        "north": max(point[1] for point in geographic),
    }
    return bbox, crs


def build_index(archive: str, splits_csv: str, db_path: str, index_path: str, checkpoint: str) -> int:
    adapter = DynamicEarthNetAdapter(archive)
    model = CLIPViTB32EmbeddingModel(checkpoint_path=checkpoint)
    vector_store = FaissVectorStore(dim=model.embedding_dim)
    database = DatabaseManager(db_path)
    database.init_db()
    tiles = TileRepository(database)
    areas = _training_area_ids(Path(splits_csv))
    missing = areas - set(adapter.areas)
    if missing:
        adapter.close()
        raise ValueError(f"Training split AOIs absent from DynamicEarthNet archive: {sorted(missing)[:5]}")
    if not areas:
        adapter.close()
        raise ValueError("No training AOIs found in split CSV")

    dates_by_location: dict[str, list[str]] = {}
    embeddings, vector_ids, indexed_areas = [], [], []
    for area_id in sorted(areas):
        location = f"den{area_id}"
        dates = adapter.list_dates(location)
        dates_by_location[location] = dates
        crs, _ = adapter.get_area_georeferencing(location)
        area_images = []
        for row in range(4):
            for col in range(4):
                tile_id = f"{location}_{row:04d}_{col:04d}_planet"
                bbox, _ = _tile_bbox(adapter, location, row, col)
                # Index the first annotated month, with imagery from the matching date.
                image = adapter.load_tile_array(location, dates[0], tile_id)
                area_images.append(image)
                vector_id = f"{tile_id}@{dates[0]}"
                vector_ids.append(vector_id)
                for acquisition_date in dates:
                    tiles.insert_tile(TileRecord(
                        tile_id=tile_id,
                        location_id=location,
                        sensor="planet",
                        date=acquisition_date,
                        bbox_json=json.dumps(bbox),
                        crs=crs,
                        resolution_m=3.0,
                        embedding_id=vector_id,
                        ingested_at=f"{acquisition_date}T00:00:00Z",
                    ))
        embeddings.extend(model.encode_images(area_images))
        indexed_areas.append(area_id)
        print(f"Indexed {location} ({len(indexed_areas)}/{len(areas)} training AOIs)", flush=True)

    vector_store.add(np.stack(embeddings), vector_ids)
    vector_store.save(index_path)
    metadata_path = Path(f"{index_path}.meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "model": "clip_vit_b32",
        "embedding_dim": model.embedding_dim,
        "indexed_date": datetime.now(timezone.utc).isoformat(),
        "dataset": "DynamicEarthNet-video 71-PSNR TACO distribution",
        "split": "train",
        "indexed_date_for_image": "first monthly annotated acquisition",
        "areas": indexed_areas,
        "tile_count": vector_store.size,
        "monthly_dates_per_tile": 24,
        "labels": ["impervious", "agriculture", "forest and other vegetation", "wetlands", "bare soil", "water", "snow and ice"],
    }, indent=2), encoding="utf-8")
    adapter.close()
    print(f"Built {vector_store.size} DynamicEarthNet training tile vectors across {len(areas)} AOIs")
    return vector_store.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip")
    parser.add_argument("--splits-csv", default="data/dynamicearthnet/splits.csv")
    parser.add_argument("--db-path", default="data/dynamicearthnet.db")
    parser.add_argument("--index-path", default="data/faiss_dynamicearthnet.index")
    parser.add_argument("--checkpoint", default="models/clip_vit_b32.pt")
    args = parser.parse_args()
    build_index(args.archive, args.splits_csv, args.db_path, args.index_path, args.checkpoint)
