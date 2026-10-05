"""Change detection service coordinating inference, postprocessing, and reporting."""
import time
import uuid
import json
import colorsys
from typing import Optional
import numpy as np
from app.models.interfaces import ChangeDetector
from app.services.image_service import ImageService
from app.services.comparison_service import ComparisonService
from app.services.summary_service import SummaryService
from app.postprocessing.mask_refinement import refine_change_mask
from app.postprocessing.bbox_extraction import extract_bounding_boxes
from app.postprocessing.confidence import compute_confidence
from app.schemas.change_detection import (
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    DetectionEvaluation,
    SemanticTransition,
)
from app.exceptions import invalid_query_exception
from app.services.temporal_state_store import TemporalStateStore
from app.services.review_service import ReviewService


LAND_COVER_CLASSES = (
    "impervious surface", "agriculture", "forest and other vegetation",
    "wetlands", "bare soil", "water", "snow and ice",
)


class ChangeDetectionService:
    """Coordinates change detection model execution, mask refinement, and analyst summary."""

    def __init__(
        self,
        change_detector: ChangeDetector,
        image_service: ImageService,
        comparison_service: ComparisonService,
        summary_service: SummaryService,
        min_change_area_px: int = 25,
        temporal_state_store: Optional[TemporalStateStore] = None,
        review_service: Optional[ReviewService] = None,
    ):
        self.change_detector = change_detector
        self.image_service = image_service
        self.comparison_service = comparison_service
        self.summary_service = summary_service
        self.min_change_area_px = min_change_area_px
        self.temporal_state_store = temporal_state_store
        self.review_service = review_service

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

        if not date_before or not date_after:
            raise invalid_query_exception("Could not resolve both comparison dates.")

        # 1. For Mamba, run all available dates in order. Pair detectors load
        # only the comparison endpoints.
        if hasattr(self.change_detector, "detect_sequence"):
            tile_dates = self.image_service.tile_repo.list_dates_for_tile(tile_id)
            if req.temporal_dates:
                sequence_dates = req.temporal_dates
                if sequence_dates != sorted(set(sequence_dates)):
                    raise invalid_query_exception("temporal_dates must be unique and sorted ascending.")
                if sequence_dates[0] != date_before or sequence_dates[-1] != date_after:
                    raise invalid_query_exception(
                        "temporal_dates must start at date_before and end at date_after."
                    )
            else:
                if getattr(self.change_detector, "transition_classes", 0):
                    # The semantic head is trained on labeled date pairs; use
                    # the requested pair and continue any saved state between them.
                    sequence_dates = sorted(set((date_before, date_after)))
                else:
                    sequence_dates = [
                        date for date in tile_dates if date_before <= date <= date_after
                    ]
                    if date_before not in sequence_dates:
                        sequence_dates.insert(0, date_before)
                    if date_after not in sequence_dates:
                        sequence_dates.append(date_after)
                    sequence_dates = sorted(set(sequence_dates))

            initial_context = None
            context_date = sequence_dates[0]
            model_fingerprint = getattr(self.change_detector, "model_fingerprint", "")
            if self.temporal_state_store and model_fingerprint:
                initial_context = self.temporal_state_store.load(
                    tile_id, context_date, model_fingerprint
                )

            if initial_context:
                processed_dates = [date for date in sequence_dates if date > context_date]
                sequence_images = [self.image_service.get_tile_array(tile_id, date) for date in processed_dates]
            else:
                processed_dates = sequence_dates
                sequence_images = [self.image_service.get_tile_array(tile_id, date) for date in processed_dates]

            # Some legacy/mock tiles have no tile-level date records; preserve
            # the explicit comparison as the minimum two-frame SSM sequence.
            if len(sequence_images) < (1 if initial_context else 2):
                initial_context = None
                processed_dates = list(dict.fromkeys([date_before, date_after]))
                sequence_images = [
                    self.image_service.get_tile_array(tile_id, date)
                    for date in processed_dates
                ]

            raw_output = self.change_detector.detect_sequence(
                sequence_images,
                initial_context=initial_context,
            )
            temporal_contexts = (raw_output.metadata or {}).get("temporal_contexts", [])
            if self.temporal_state_store and model_fingerprint:
                for date, context in zip(processed_dates, temporal_contexts):
                    self.temporal_state_store.save(
                        tile_id, date, model_fingerprint, context
                    )
        else:
            # 2. Non-temporal baselines keep their existing pair interface.
            arr_before = self.image_service.get_tile_array(tile_id, date_before)
            arr_after = self.image_service.get_tile_array(tile_id, date_after)
            raw_output = self.change_detector.detect(arr_before, arr_after)

        # 3. Postprocess mask (morphological opening/closing)
        refined_mask = refine_change_mask(raw_output.mask, min_area_px=self.min_change_area_px)

        # 4. Extract bounding boxes
        boxes = extract_bounding_boxes(
            refined_mask,
            min_area_px=self.min_change_area_px,
            base_confidence=raw_output.confidence_score
        )
        evaluation = self.image_service.evaluate_change_mask(
            tile_id, location_id, date_before, date_after, refined_mask
        )

        job_id = str(uuid.uuid4())
        semantic_mask = (raw_output.metadata or {}).get("semantic_mask")
        semantic_confidence = (raw_output.metadata or {}).get("semantic_confidence")
        semantic_transitions: list[SemanticTransition] = []
        semantic_mask_url = None
        if semantic_mask is not None:
            semantic_mask = np.asarray(semantic_mask, dtype=np.uint8).copy()
            if semantic_mask.shape == refined_mask.shape:
                semantic_mask[refined_mask == 0] = 0
                counts = np.bincount(semantic_mask.ravel(), minlength=50)
                confidence_map = (
                    np.asarray(semantic_confidence, dtype=np.float32)
                    if semantic_confidence is not None
                    else np.full(semantic_mask.shape, raw_output.confidence_score, dtype=np.float32)
                )
                total_pixels = max(int(semantic_mask.size), 1)
                for encoded_id in np.flatnonzero(counts[1:]) + 1:
                    transition_pixels = semantic_mask == encoded_id
                    class_pair = int(encoded_id) - 1
                    semantic_transitions.append(SemanticTransition(
                        from_class=LAND_COVER_CLASSES[class_pair // 7],
                        to_class=LAND_COVER_CLASSES[class_pair % 7],
                        pixel_count=int(counts[encoded_id]),
                        area_fraction=float(counts[encoded_id] / total_pixels),
                        model_score=float(np.mean(confidence_map[transition_pixels])),
                        color="#{:02x}{:02x}{:02x}".format(*(
                            np.round(np.asarray(colorsys.hsv_to_rgb(
                                (class_pair * 0.61803398875) % 1.0, 0.82, 1.0
                            )) * 255).astype(np.uint8).tolist()
                        )),
                        bounding_boxes=extract_bounding_boxes(
                            transition_pixels.astype(np.uint8),
                            min_area_px=self.min_change_area_px,
                            base_confidence=float(np.mean(confidence_map[transition_pixels])),
                        ),
                    ))
                semantic_transitions.sort(key=lambda item: item.pixel_count, reverse=True)
                semantic_filename = self.image_service.save_semantic_mask_bytes(job_id, semantic_mask)
                semantic_mask_url = f"/api/v1/masks/{semantic_filename}"

        # 5. Save mask PNG file
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

        response = ChangeDetectionResponse(
            job_id=job_id,
            status="completed",
            mask_url=mask_url,
            semantic_mask_url=semantic_mask_url,
            semantic_transitions=semantic_transitions,
            bounding_boxes=boxes,
            summary=summary,
            evaluation=DetectionEvaluation(**evaluation) if evaluation else None,
            processing_ms=max(processing_ms, 1)
        )
        if self.review_service:
            record = self.image_service.tile_repo.get_by_id_and_date(tile_id, date_after)
            try:
                bbox = json.loads(record.bbox_json) if record else None
            except (TypeError, ValueError):
                bbox = None
            self.review_service.save_candidate(
                job_id=job_id,
                tile_id=tile_id,
                location_id=location_id,
                date_before=date_before,
                date_after=date_after,
                candidate=response.model_dump(mode="json"),
                provenance={
                    "tile_id": tile_id,
                    "location_id": location_id,
                    "date_before": date_before,
                    "date_after": date_after,
                    "sensor": record.sensor if record else "unknown",
                    "crs": record.crs if record else None,
                    "resolution_m": record.resolution_m if record else None,
                    "geo_bbox": bbox,
                    "detector": raw_output.detector_name,
                    "processing_ms": response.processing_ms,
                    "mask_url": mask_url,
                },
            )
        return response
