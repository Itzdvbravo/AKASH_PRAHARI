"""Comparison request and response schemas."""
from typing import Optional
from pydantic import BaseModel, Field
from .common import SensorType, TileRef


class ComparisonRequest(BaseModel):
    location_id: str
    tile_id: str
    date_before: str
    date_after: str
    sensor: SensorType = SensorType.SENTINEL_2


class ComparisonTileInfo(BaseModel):
    tile_ref: TileRef
    image_url: str
    cloud_cover_pct: Optional[float] = None


class ComparisonResponse(BaseModel):
    comparison_id: str
    before: ComparisonTileInfo
    after: ComparisonTileInfo
    coregistered: bool = True
