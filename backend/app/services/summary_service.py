"""Summary service assembling structured analyst intelligence reports."""
from typing import Optional
from app.schemas.change_detection import AnalystSummary
from app.schemas.common import Confidence
from app.models.interfaces import ChangeDetectionOutput


class SummaryService:
    """Assembles structured analyst summaries from change detection outputs."""

    def build_summary(
        self,
        location_id: str,
        date_before: str,
        date_after: str,
        cd_output: ChangeDetectionOutput,
        detector_name: str = "pixel_diff_otsu"
    ) -> AnalystSummary:
        # Binary OSCD masks do not identify what land-cover class changed.
        # Avoid inventing semantic classes from the fraction of changed pixels.
        change_type = "binary_surface_change" if cd_output.changed_pixel_fraction > 0 else "no_change_detected"

        confidence_obj = Confidence(
            score=cd_output.confidence_score,
            method=f"{detector_name}_uncalibrated_score",
            calibrated=False
        )

        return AnalystSummary(
            location=location_id.capitalize(),
            date_before=date_before,
            date_after=date_after,
            change_type=change_type,
            earliest_detectable_change=None,
            confidence=confidence_obj,
            changed_pixel_fraction=cd_output.changed_pixel_fraction,
            source_provenance=f"Sentinel-2 / OSCD ({detector_name})",
            detector=detector_name
        )
