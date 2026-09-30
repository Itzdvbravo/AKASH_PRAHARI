"""Change detection service coordinating inference, postprocessing, and reporting."""
import time
import uuid
from typing import Optional
from app.models.interfaces import ChangeDetector
from app.services.image_service import ImageService
from app.services.comparison_service import ComparisonService
from app.services.summary_service import SummaryService
from app.postprocessing.mask_refinement import refine_change_mask
from app.postprocessing.bbox_extraction import extract_bounding_boxes
from app.postprocessing.confidence import compute_confidence
from app.schemas.change_detection import ChangeDetectionRequest, ChangeDetectionResponse
from app.exceptions import invalid_query_exception


class ChangeDetectionService:
    """Coordinates change detection model execution, mask refinement, and analyst summary."""

    def __init__(
        self,
        change_detector: ChangeDetector,
        image_service: ImageService,
        comparison_service: ComparisonService,
        summary_service: SummaryService,
        min_change_area_px: int = 25
    ):
        self.change_detector = change_detector
        self.image_service = image_service
        self.comparison_service = comparison_service
        self.summary_service = summary_service
        self.min_change_area_px = min_change_area_px

    def run_detection(self, req: ChangeDetectionRequest) -> ChangeDetectionResponse:
        start_t = time.perf_counter()

        # Resolve request parameters
        location_id = req.location_id
        tile_id = req.tile_id
        date_before = req.date_before
        date_after = req.date_after

        if req.comparison_id:
            comp = self.comparison_service.get_comparison_request(req.comparison_id)
            if comp:
                location_id = comp.location_id
                tile_id = comp.tile_id
                date_before = comp.date_before
                date_after = comp.date_after

        if not location_id or not tile_id:
            raise invalid_query_exception("Missing required change detection parameters (tile_id, location_id).")

        if not date_before or not date_after:
            dates = self.image_service.get_available_dates(location_id)
            if len(dates) >= 2:
                date_before = date_before or dates[0]
                date_after = date_after or dates[1]
            else:
                date_before = date_before or "2016-03-15"
                date_after = date_after or "2018-06-20"

        # 1. Fetch bi-temporal image tile arrays
        arr_before = self.image_service.get_tile_array(tile_id, date_before)
        arr_after = self.image_service.get_tile_array(tile_id, date_after)

        # 2. Run change detector
        raw_output = self.change_detector.detect(arr_before, arr_after)

        # 3. Postprocess mask (morphological opening/closing)
        refined_mask = refine_change_mask(raw_output.mask, min_area_px=self.min_change_area_px)

        # 4. Extract bounding boxes
        boxes = extract_bounding_boxes(
            refined_mask,
            min_area_px=self.min_change_area_px,
            base_confidence=raw_output.confidence_score
        )

        # 5. Save mask PNG file
        job_id = str(uuid.uuid4())
        mask_filename = self.image_service.save_mask_bytes(job_id, refined_mask)
        mask_url = f"/api/v1/masks/{mask_filename}"

        # 6. Build Analyst Summary
        summary = self.summary_service.build_summary(
            location_id=location_id,
            date_before=date_before,
            date_after=date_after,
            cd_output=raw_output,
            detector_name=raw_output.detector_name
        )

        processing_ms = int((time.perf_counter() - start_t) * 1000)

        return ChangeDetectionResponse(
            job_id=job_id,
            status="completed",
            mask_url=mask_url,
            bounding_boxes=boxes,
            summary=summary,
            processing_ms=max(processing_ms, 1)
        )
