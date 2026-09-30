"""Image metadata and dates request/response schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field
from .common import SensorType, GeoBBox


class DatesResponse(BaseModel):
    location_id: str
    dates: List[str]
    sensor: SensorType = SensorType.SENTINEL_2


class ImageMetadataResponse(BaseModel):
    tile_id: str
    location_id: str
    date: str
    sensor: SensorType
    width: int
    height: int
    channels: int
    crs: str
    geo_bbox: GeoBBox
