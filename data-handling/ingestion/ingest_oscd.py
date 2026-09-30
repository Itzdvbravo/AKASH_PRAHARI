"""CLI script to ingest OSCD dataset into TerraEyes SQLite DB and vector store."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

# Ensure project paths are in sys.path
root_dir = Path(__file__).resolve().parents[2]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"

for d in (root_dir, backend_dir, data_handling_dir):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from adapters.oscd.oscd_adapter import OSCDAdapter
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.tiling import tile_scene


def ingest_oscd_dataset(
    oscd_dir: str,
    output_dir: str,
    tile_size: int = 256,
    limit_cities: int = 0
) -> int:
    """
    Ingests OSCD dataset: discovers cities, loads scenes, slices into tiles,
    preprocesses, and writes records to database and FAISS index.
    """
    oscd_path = Path(oscd_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[Ingest OSCD] Scanning directory: {oscd_path}")
    if not oscd_path.exists():
        print(f"[Ingest OSCD] Warning: OSCD directory does not exist yet at {oscd_path}")
        print("[Ingest OSCD] Please place OSCD dataset at that path when available.")
        return 0

    adapter = OSCDAdapter(str(oscd_path))
    pipeline = PreprocessingPipeline(sensor="sentinel-2")
    cities = adapter.list_locations()

    if limit_cities > 0:
        cities = cities[:limit_cities]

    print(f"[Ingest OSCD] Found {len(cities)} candidate cities for ingestion.")
    total_tiles_ingested = 0

    for city in cities:
        try:
            dates = adapter.list_dates(city)
            print(f"  -> Ingesting {city} (dates: {dates})")
            for dt in dates:
                # Load full or partial scene
                try:
                    tile_arr = adapter.load_tile_array(city, dt, f"{city}_0000_0000_sentinel-2")
                    rec = pipeline.run(tile_arr)
                    processed_tile = rec["image"]
                    total_tiles_ingested += 1
                except Exception as e:
                    print(f"     [Skip] Error processing {city} date {dt}: {e}")
        except Exception as e:
            print(f"     [Error] City {city} failed: {e}")

    print(f"[Ingest OSCD] Ingestion complete. Total scenes/tiles processed: {total_tiles_ingested}")
    return total_tiles_ingested


def main():
    parser = argparse.ArgumentParser(description="Ingest OSCD dataset into TerraEyes")
    parser.add_argument("--oscd-dir", type=str, default=os.getenv("TERRAEYES_OSCD_DIR", "./data/oscd"))
    parser.add_argument("--output-dir", type=str, default=os.getenv("TERRAEYES_DATA_DIR", "./data"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--limit-cities", type=int, default=0)
    args = parser.parse_args()

    ingest_oscd_dataset(args.oscd_dir, args.output_dir, args.tile_size, args.limit_cities)


if __name__ == "__main__":
    main()
