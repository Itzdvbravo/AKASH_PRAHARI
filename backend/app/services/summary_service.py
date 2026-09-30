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
        # Infer semantic category based on changed fraction & detector signal
        pct = cd_output.changed_pixel_fraction
        if pct > 0.15:
            change_type = "urban_expansion_or_earthworks"
        elif pct > 0.05:
            change_type = "structural_development"
        elif pct > 0.01:
            change_type = "minor_surface_alteration"
        else:
            change_type = "negligible_variation"

        confidence_obj = Confidence(
            score=cd_output.confidence_score,
            method="pixel_fraction_and_diff_magnitude",
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
