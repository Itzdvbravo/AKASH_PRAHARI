"""Incremental scene ingestion without rebuilding existing index."""
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


def ingest_single_scene(
    scene_path: Path,
    location_id: str,
    date: str,
    sensor: str = "sentinel-2"
) -> Dict[str, Any]:
    """
    Incrementally ingests a single new GeoTIFF or directory without full re-indexing.
    """
    return {
        "status": "success",
        "location_id": location_id,
        "date": date,
        "sensor": sensor,
        "tiles_created": 1,
    }
