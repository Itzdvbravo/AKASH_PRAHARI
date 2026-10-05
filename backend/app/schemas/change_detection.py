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
    source_provenance: str = "Planet / DynamicEarthNet"
    detector: str = "pixel_diff"


class ChangeDetectionRequest(BaseModel):
    comparison_id: Optional[str] = None
    location_id: Optional[str] = None
    tile_id: Optional[str] = None
    date_before: Optional[str] = None
    date_after: Optional[str] = None
    temporal_dates: Optional[List[str]] = Field(
        default=None,
        description="Optional ordered acquisitions to process for an SSM change comparison",
        min_length=2,
    )


class DetectionEvaluation(BaseModel):
    precision: float
    recall: float
    f1: float
    iou: float
    pixel_accuracy: float
    specificity: float
    balanced_accuracy: float
    predicted_changed_pixels: int
    expected_changed_pixels: int
    pixel_count: int
    source: str


class SemanticTransition(BaseModel):
    from_class: str
    to_class: str
    pixel_count: int
    area_fraction: float
    model_score: float
    color: str
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)


class ChangeDetectionResponse(BaseModel):
    job_id: str
    status: str = "completed"
    mask_url: str
    semantic_mask_url: Optional[str] = None
    semantic_transitions: List[SemanticTransition] = Field(default_factory=list)
    bounding_boxes: List[BoundingBox]
    summary: AnalystSummary
    evaluation: Optional[DetectionEvaluation] = None
    processing_ms: int
