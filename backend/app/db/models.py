"""Domain and DB row models."""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TileRecord:
    tile_id: str
    location_id: str
    sensor: str
    date: str
    bbox_json: str
    crs: str = "EPSG:4326"
    resolution_m: float = 10.0
    embedding_id: Optional[str] = None
    ingested_at: Optional[str] = None
