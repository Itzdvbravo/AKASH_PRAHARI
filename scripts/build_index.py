"""Build the offline CLIP image index from preprocessed OSCD image pairs."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"
for directory in (root_dir, backend_dir, data_handling_dir):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from app.db.connection import DatabaseManager
from app.db.models import TileRecord
from app.db.repositories.tile_repo import TileRepository
from app.models.embedding.clip_vit_b32 import CLIPViTB32EmbeddingModel
from app.vector_store.faiss_store import FaissVectorStore
from adapters.oscd.oscd_metadata import OSCDMetadataExtractor


def build_vector_index(
    manifest_path: str,
    oscd_dir: str,
    db_path: str,
    index_output_path: str,
    checkpoint_path: str,
    splits_path: str,
) -> int:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    allowed_cities = set(
        json.loads(Path(splits_path).read_text(encoding="utf-8"))["train"]
    )
    model = CLIPViTB32EmbeddingModel(checkpoint_path=checkpoint_path)
    vector_store = FaissVectorStore(dim=model.embedding_dim)
    db_manager = DatabaseManager(db_path)
    db_manager.init_db()
    tile_repo = TileRepository(db_manager)
    metadata = OSCDMetadataExtractor(Path(oscd_dir))

    embeddings = []
    tile_ids = []
    seen_ids = set()
    indexed_cities = set()
    for tile in manifest.get("tiles", []):
        city = tile["location_id"]
        if city not in allowed_cities:
            continue
        tile_id = tile["tile_id"]
        if tile_id in seen_ids:
            raise ValueError(f"Duplicate tile ID in ingestion manifest: {tile_id}")
        array_path = manifest_file.parent / tile["array_path"]
        with np.load(array_path, allow_pickle=False) as arrays:
            image = arrays["after"].astype(np.float32)
        embedding = model.encode_image(image)
        if embedding.shape != (model.embedding_dim,):
            raise ValueError(f"Unexpected embedding shape for {tile_id}: {embedding.shape}")

        bbox_json = json.dumps(metadata.get_geo_bbox(city))
        ingested_at = datetime.now(timezone.utc).isoformat()
        for date in (tile["date_before"], tile["date_after"]):
            tile_repo.insert_tile(TileRecord(
                tile_id=tile_id,
                location_id=city,
                sensor="sentinel-2",
                date=date,
                bbox_json=bbox_json,
                resolution_m=10.0,
                embedding_id=tile_id,
                ingested_at=ingested_at,
            ))
        embeddings.append(embedding)
        tile_ids.append(tile_id)
        seen_ids.add(tile_id)
        indexed_cities.add(city)

    if not embeddings:
        raise ValueError(
            "No training tiles were found. Run scripts/ingest_oscd.py against "
            "the local OSCD archive and verify the train split."
        )

    vector_store.add(np.stack(embeddings), tile_ids)
    vector_store.save(index_output_path)
    index_metadata = {
        "model": "clip_vit_b32",
        "embedding_dim": model.embedding_dim,
        "indexed_date": datetime.now(timezone.utc).isoformat(),
        "cities": sorted(indexed_cities),
        "tile_count": vector_store.size,
        "split": "train",
        "image_date": "after",
    }
    metadata_file = Path(f"{index_output_path}.meta.json")
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(json.dumps(index_metadata, indent=2), encoding="utf-8")
    print(
        f"Indexed {vector_store.size} training tiles from {len(indexed_cities)} cities "
        f"to {index_output_path}"
    )
    return vector_store.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the offline CLIP image index")
    parser.add_argument("--manifest", default="./data/oscd_tiles/manifest.json")
    parser.add_argument("--oscd-dir", default="./images")
    parser.add_argument("--db-path", default="./data/terraeyes.db")
    parser.add_argument("--index-path", default="./data/faiss.index")
    parser.add_argument("--checkpoint-path", default="./models/clip_vit_b32.pt")
    parser.add_argument("--splits", default="./data-handling/splits/oscd_splits.json")
    arguments = parser.parse_args()
    build_vector_index(
        arguments.manifest,
        arguments.oscd_dir,
        arguments.db_path,
        arguments.index_path,
        arguments.checkpoint_path,
        arguments.splits,
    )
