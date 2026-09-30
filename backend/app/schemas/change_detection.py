"""Change detection request and analyst summary schemas."""
from typing import List, Optional
from pydantic import BaseModel, Field
from .common import BoundingBox, Confidence


class AnalystSummary(BaseModel):
    location: str
    date_before: str
    date_after: str
    change_type: str = "surface_alteration"
    earliest_detectable_change: Optional[str] = None
    confidence: Confidence
    changed_pixel_fraction: float
    source_provenance: str = "Sentinel-2 / OSCD"
    detector: str = "pixel_diff"


class ChangeDetectionRequest(BaseModel):
    comparison_id: Optional[str] = None
    location_id: Optional[str] = None
    tile_id: Optional[str] = None
    date_before: Optional[str] = None
    date_after: Optional[str] = None


class ChangeDetectionResponse(BaseModel):
    job_id: str
    status: str = "completed"
    mask_url: str
    bounding_boxes: List[BoundingBox]
    summary: AnalystSummary
    processing_ms: int
