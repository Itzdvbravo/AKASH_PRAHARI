"""Script to build and save offline FAISS vector store index from dataset."""
import argparse
from pathlib import Path
import sys
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"
for p in (root_dir, backend_dir, data_handling_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.vector_store.faiss_store import FaissVectorStore
from app.models.embedding.mock_embedding import MockEmbeddingModel
from app.db.connection import DatabaseManager
from app.db.repositories.tile_repo import TileRepository
from adapters.oscd.oscd_metadata import OSCD_CITY_COORDINATES


def build_vector_index(db_path: str, index_output_path: str, dim: int = 512):
    print(f"[Build Index] Connecting to DB: {db_path}")
    db_mgr = DatabaseManager(db_path)
    db_mgr.init_db()
    tile_repo = TileRepository(db_mgr)

    model = MockEmbeddingModel(dim=dim)
    store = FaissVectorStore(dim=dim)

    locations = tile_repo.list_locations()
    if not locations:
        print("[Build Index] No tiles in database. Using predefined OSCD cities.")
        locations = list(OSCD_CITY_COORDINATES.keys())

    print(f"[Build Index] Generating embeddings for {len(locations)} locations...")
    embeddings = []
    ids = []
    for loc in locations:
        tile_id = f"{loc}_0001_0001_sentinel-2"
        text = f"satellite view of {loc} urban infrastructure"
        vec = model.encode_text(text)
        embeddings.append(vec)
        ids.append(tile_id)

    store.add(np.array(embeddings), ids)
    store.save(index_output_path)
    print(f"[Build Index] Saved FAISS vector index with {store.size} entries to: {index_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline vector search index")
    parser.add_argument("--db-path", type=str, default="./data/terraeyes.db")
    parser.add_argument("--index-path", type=str, default="./data/faiss.index")
    parser.add_argument("--dim", type=int, default=512)
    args = parser.parse_args()
    build_vector_index(args.db_path, args.index_path, args.dim)
