"""Comparison service coordinating bi-temporal tile pairs."""
import uuid
from typing import Dict, Optional
from app.schemas.comparison import ComparisonRequest, ComparisonResponse, ComparisonTileInfo
from app.schemas.common import TileRef, SensorType
from app.services.image_service import ImageService
from app.geospatial.alignment import check_bitemporal_alignment


class ComparisonService:
    """Manages creation and verification of bi-temporal comparisons."""

    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        self._comparisons_store: Dict[str, ComparisonRequest] = {}

    def create_comparison(self, req: ComparisonRequest) -> ComparisonResponse:
        comparison_id = str(uuid.uuid4())
        self._comparisons_store[comparison_id] = req

        # Load tile arrays to check alignment
        tile_a = self.image_service.get_tile_array(req.tile_id, req.date_before)
        tile_b = self.image_service.get_tile_array(req.tile_id, req.date_after)
        is_aligned = check_bitemporal_alignment(tile_a, tile_b)

        before_info = ComparisonTileInfo(
            tile_ref=TileRef(
                tile_id=req.tile_id,
                location_id=req.location_id,
                date=req.date_before,
                sensor=req.sensor
            ),
            image_url=f"/api/v1/images/{req.tile_id}/{req.date_before}",
            cloud_cover_pct=None
        )

        after_info = ComparisonTileInfo(
            tile_ref=TileRef(
                tile_id=req.tile_id,
                location_id=req.location_id,
                date=req.date_after,
                sensor=req.sensor
            ),
            image_url=f"/api/v1/images/{req.tile_id}/{req.date_after}",
            cloud_cover_pct=None
        )

        return ComparisonResponse(
            comparison_id=comparison_id,
            before=before_info,
            after=after_info,
            coregistered=is_aligned
        )

    def get_comparison_request(self, comparison_id: str) -> Optional[ComparisonRequest]:
        return self._comparisons_store.get(comparison_id)
