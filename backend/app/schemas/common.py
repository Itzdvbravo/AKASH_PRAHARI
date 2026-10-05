"""Common Pydantic schemas and domain types shared across API endpoints."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SensorType(str, Enum):
    PLANET = "planet"
    LANDSAT = "landsat"
    SENTINEL_1 = "sentinel-1"
    SENTINEL_2 = "sentinel-2"
    UNKNOWN = "unknown"
    ANY = "any"


class GeoBBox(BaseModel):
    west: float = Field(..., description="Westernmost longitude")
    south: float = Field(..., description="Southernmost latitude")
    east: float = Field(..., description="Easternmost longitude")
    north: float = Field(..., description="Northernmost latitude")


class Confidence(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    method: str = Field(..., description="Calculation method (e.g. cosine_similarity, pixel_fraction)")
    calibrated: bool = Field(default=False, description="Whether score is statistically calibrated")


class TileRef(BaseModel):
    tile_id: str
    location_id: str
    date: str
    sensor: SensorType = SensorType.SENTINEL_2


class BoundingBox(BaseModel):
    x: int = Field(..., description="X coordinate (pixels from left)")
    y: int = Field(..., description="Y coordinate (pixels from top)")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")
    label: str = Field(default="change", description="Semantic label")
    confidence: Confidence
    geo_bbox: Optional[GeoBBox] = None
