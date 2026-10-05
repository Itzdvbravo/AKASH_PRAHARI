"""Preprocess and persist aligned OSCD image-pair tiles for later indexing."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

# Ensure project paths are importable when this file is run as a CLI.
root_dir = Path(__file__).resolve().parents[2]
data_handling_dir = root_dir / "data-handling"
for directory in (root_dir, data_handling_dir):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from adapters.oscd.oscd_adapter import OSCDAdapter
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.tiling import tile_scene


def ingest_oscd_dataset(
    oscd_dir: str,
    output_dir: str,
    tile_size: int = 256,
    limit_cities: int = 0,
) -> int:
    """Preprocess each acquisition pair and save corresponding fixed-grid tiles.

    Embedding/index creation is a later model-approved step. This command
    persists reproducible image arrays and their provenance in a JSON manifest.
    """
    oscd_path = Path(oscd_dir)
    out_path = Path(output_dir)
    if not oscd_path.is_dir():
        print(f"[Ingest OSCD] Dataset directory not found: {oscd_path}")
        return 0

    adapter = OSCDAdapter(str(oscd_path))
    pipeline = PreprocessingPipeline(sensor="sentinel-2")
    cities = adapter.list_locations()
    if limit_cities > 0:
        cities = cities[:limit_cities]
    print(f"[Ingest OSCD] Found {len(cities)} candidate cities for ingestion.")

    manifest = {
        "dataset": "OSCD",
        "source_dir": str(oscd_path.resolve()),
        "tile_size": tile_size,
        "sensor": "sentinel-2",
        "tiles": [],
    }
    total_tiles = 0

    for city in cities:
        try:
            dates = adapter.list_dates(city)
            if len(dates) != 2:
                raise ValueError(f"Expected two acquisition dates, found {dates}")
            city_dir = adapter._get_cities()[city]
            before_raw = adapter.loader.load_band_composite(city_dir, time_index=1)
            after_raw = adapter.loader.load_band_composite(city_dir, time_index=2)
            if before_raw.shape != after_raw.shape:
                raise ValueError(
                    f"Rectified scenes differ in shape: {before_raw.shape} vs {after_raw.shape}"
                )

            # OSCD's *_rect products are already co-registered; retain the
            # pair on one grid and apply the same deterministic preprocessing.
            before = pipeline.run(before_raw)["image"]
            after = pipeline.run(after_raw)["image"]
            before_tiles = tile_scene(before, tile_size=tile_size, location_id=city)
            after_tiles = tile_scene(after, tile_size=tile_size, location_id=city)
            if [tile[0] for tile in before_tiles] != [tile[0] for tile in after_tiles]:
                raise ValueError("Before/after tile grids do not match")

            city_out = out_path / "oscd_tiles" / city
            city_out.mkdir(parents=True, exist_ok=True)
            for (tile_id, before_tile, bounds), (_, after_tile, _) in zip(
                before_tiles, after_tiles
            ):
                tile_path = city_out / f"{tile_id}.npz"
                np.savez_compressed(
                    tile_path,
                    before=before_tile.astype(np.float32),
                    after=after_tile.astype(np.float32),
                    date_before=dates[0],
                    date_after=dates[1],
                )
                manifest["tiles"].append({
                    "tile_id": tile_id,
                    "location_id": city,
                    "date_before": dates[0],
                    "date_after": dates[1],
                    "bounds_pixels": list(bounds),
                    "array_path": str(tile_path.relative_to(out_path / "oscd_tiles")),
                    "shape": list(before_tile.shape),
                    "bands": ["B04", "B03", "B02"],
                    "source_coregistered": True,
                })
                total_tiles += 1
            print(f"  -> {city}: wrote {len(before_tiles)} aligned tiles")
        except Exception as exc:
            print(f"  -> {city}: failed: {exc}")

    manifest_path = out_path / "oscd_tiles" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Ingest OSCD] Wrote {total_tiles} tiles. Manifest: {manifest_path}")
    return total_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and tile OSCD image pairs")
    parser.add_argument("--oscd-dir", default=os.getenv("TERRAEYES_OSCD_DIR", "./images"))
    parser.add_argument("--output-dir", default=os.getenv("TERRAEYES_DATA_DIR", "./data"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--limit-cities", type=int, default=0)
    args = parser.parse_args()
    if args.tile_size <= 0:
        parser.error("--tile-size must be positive")
    ingest_oscd_dataset(args.oscd_dir, args.output_dir, args.tile_size, args.limit_cities)


if __name__ == "__main__":
    main()
