"""Unit tests for services: retrieval, comparison, change detection, summary."""
import json
import numpy as np
from app.db.models import TileRecord
from app.services.image_service import ImageService
from app.services.comparison_service import ComparisonService
from app.services.summary_service import SummaryService
from app.services.change_detection_service import ChangeDetectionService
from app.services.retrieval_service import RetrievalService
from app.services.query_orchestrator import QueryOrchestrator
from app.schemas.search import SearchRequest
from app.schemas.comparison import ComparisonRequest
from app.schemas.change_detection import ChangeDetectionRequest


def test_retrieval_service_flow(tile_repo, vector_store, mock_embedding, tmp_path):
    # Add dummy tile
    rec = TileRecord(
        tile_id="paris_0001_0001_sentinel-2",
        location_id="paris",
        sensor="sentinel-2",
        date="2018-05-10",
        bbox_json=json.dumps({"west": 2.2, "south": 48.8, "east": 2.4, "north": 48.9}),
        crs="EPSG:4326",
        resolution_m=10.0,
        embedding_id="paris_0001_0001_sentinel-2"
    )
    tile_repo.insert_tile(rec)

    emb = mock_embedding.encode_text("paris scene")
    vector_store.add(np.array([emb]), ["paris_0001_0001_sentinel-2"])

    retrieval_svc = RetrievalService(mock_embedding, vector_store, tile_repo)
    orchestrator = QueryOrchestrator(retrieval_svc)

    res = orchestrator.search(SearchRequest(query_type="text", query_text="urban area in paris"))
    assert res.total >= 1
    assert res.results[0].tile_ref.location_id == "paris"
    assert res.results[0].confidence.score >= 0.0


def test_comparison_and_change_detection(tile_repo, pixel_diff_detector, tmp_path):
    masks_dir = tmp_path / "masks"
    image_svc = ImageService(tile_repo, oscd_dir=str(tmp_path / "oscd"), masks_cache_dir=str(masks_dir))
    comp_svc = ComparisonService(image_svc)
    summary_svc = SummaryService()
    cd_svc = ChangeDetectionService(pixel_diff_detector, image_svc, comp_svc, summary_svc)

    comp_res = comp_svc.create_comparison(ComparisonRequest(
        location_id="paris",
        tile_id="paris_0001_0001_sentinel-2",
        date_before="2016-03-15",
        date_after="2018-06-20"
    ))
    assert comp_res.comparison_id is not None
    assert comp_res.before.image_url.startswith("/api/v1/images/")

    cd_res = cd_svc.run_detection(ChangeDetectionRequest(
        comparison_id=comp_res.comparison_id
    ))
    assert cd_res.status == "completed"
    assert cd_res.mask_url.startswith("/api/v1/masks/")
    assert cd_res.summary.location == "Paris"
    assert cd_res.summary.confidence.score > 0.0
