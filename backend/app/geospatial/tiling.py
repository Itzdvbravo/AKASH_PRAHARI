"""Backend tile coordinate math and tile ID utilities."""
from typing import Dict


def parse_tile_id(tile_id: str) -> Dict[str, str]:
    parts = tile_id.split("_")
    if len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit():
        return {
            "location_id": parts[0],
            "row": parts[1],
            "col": parts[2],
            "sensor": parts[3],
        }
    return {
        "location_id": parts[0] if parts else "unknown",
        "row": "0000",
        "col": "0000",
        "sensor": parts[-1] if len(parts) > 1 else "unknown",
    }


def format_tile_id(location_id: str, row: int, col: int, sensor: str = "sentinel-2") -> str:
    return f"{location_id.lower()}_{row:04d}_{col:04d}_{sensor.lower()}"
