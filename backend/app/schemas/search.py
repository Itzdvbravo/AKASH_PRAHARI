"""Search request and response schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from .common import SensorType, GeoBBox, Confidence, TileRef


class SearchFilters(BaseModel):
    location: Optional[str] = Field(default=None, description="Partial or exact location/city filter")
    date_from: Optional[str] = Field(default=None, description="Start date ISO 8601")
    date_to: Optional[str] = Field(default=None, description="End date ISO 8601")
    sensor: Optional[SensorType] = Field(default=SensorType.ANY, description="Sensor filter")
    area_of_interest: Optional[GeoBBox] = Field(
        default=None, description="Optional WGS84 bounding box; intersecting tiles are retained"
    )


class SearchRequest(BaseModel):
    query_type: str = Field(default="text", description="'text' or 'image'")
    query_text: Optional[str] = Field(default="", max_length=500)
    query_image_b64: Optional[str] = Field(default=None, description="Base64 encoded query image bytes")
    filters: Optional[SearchFilters] = Field(default_factory=SearchFilters)
    top_k: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_query_payload(self):
        if self.query_type == "image" and not self.query_image_b64:
            raise ValueError("query_image_b64 must be provided when query_type is 'image'")
        if self.filters and self.filters.date_from and self.filters.date_to:
            if self.filters.date_from > self.filters.date_to:
                raise ValueError("date_from must be less than or equal to date_to")
        return self


class SearchResultItem(BaseModel):
    rank: int
    tile_ref: TileRef
    confidence: Confidence
    thumbnail_url: str
    available_dates: List[str]
    geo_bbox: GeoBBox


class SearchResponse(BaseModel):
    query_id: str
    results: List[SearchResultItem]
    total: int
    query_ms: int
