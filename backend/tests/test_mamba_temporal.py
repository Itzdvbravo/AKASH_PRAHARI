"""Coverage for temporal SSM inference and persisted recurrent state."""
from pathlib import Path

import numpy as np
import torch

from app.models.change_detection.mamba_model import MambaTemporalChangeNet
from app.models.interfaces import ChangeDetectionOutput
from app.db.models import TileRecord
from app.services.change_detection_service import ChangeDetectionService
from app.services.comparison_service import ComparisonService
from app.services.image_service import ImageService
from app.services.summary_service import SummaryService
from app.services.temporal_state_store import TemporalStateStore
from app.schemas.change_detection import ChangeDetectionRequest
from app.schemas.comparison import ComparisonRequest


def test_selective_temporal_model_accepts_multiple_dates_and_streams_incrementally():
    torch.manual_seed(7)
    model = MambaTemporalChangeNet(feature_dim=16, state_dim=4).eval()
    images = torch.rand(1, 3, 3, 32, 40)

    with torch.inference_mode():
        _pair_logits, states, conv_states, spatial, temporal = model(
            images[:, :2], return_context=True
        )
        streamed_logits = model(
            images[:, 1:],
            initial_state=states[0],
            initial_conv_state=conv_states[0],
            previous_spatial=spatial[0],
            previous_temporal=temporal[0],
        )
        three_date_logits = model(images)

    assert streamed_logits.shape == (1, 1, 32, 40)
    assert three_date_logits.shape == streamed_logits.shape
    torch.testing.assert_close(streamed_logits, three_date_logits, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(three_date_logits).all()


def test_temporal_state_store_round_trips_and_isolates_model_versions(tmp_path):
    store = TemporalStateStore(str(tmp_path / "temporal_states.h5"))
    context = {
        "state": np.ones((8, 4, 4, 5), dtype=np.float32),
        "conv": np.full((8, 2, 4, 5), 2.0, dtype=np.float32),
        "spatial": np.full((16, 4, 5), 3.0, dtype=np.float32),
        "temporal": np.full((16, 4, 5), 4.0, dtype=np.float32),
    }

    store.save("tile/a", "2020-01-01", "model-a", context)

    loaded = store.load("tile/a", "2020-01-01", "model-a")
    assert loaded is not None
    for field in context:
        np.testing.assert_allclose(loaded[field], context[field], atol=1e-3)
    assert store.load("tile/a", "2020-01-01", "model-b") is None
    assert store.load("different-tile", "2020-01-01", "model-a") is None


def test_change_service_processes_and_persists_all_tile_dates(tile_repo, tmp_path):
    tile_id = "paris_0001_0001_sentinel-2"
    dates = ["2016-03-15", "2017-06-20", "2018-06-20"]
    for date in dates:
        tile_repo.insert_tile(TileRecord(
            tile_id=tile_id,
            location_id="paris",
            sensor="sentinel-2",
            date=date,
            bbox_json='{"west":2.2,"south":48.8,"east":2.4,"north":48.9}',
            crs="EPSG:4326",
            resolution_m=10.0,
            embedding_id=tile_id,
        ))

    image_service = ImageService(
        tile_repo,
        oscd_dir=str(tmp_path / "missing-oscd"),
        masks_cache_dir=str(tmp_path / "masks"),
    )
    comparison_service = ComparisonService(image_service)
    comparison = comparison_service.create_comparison(ComparisonRequest(
        location_id="paris",
        tile_id=tile_id,
        date_before=dates[0],
        date_after=dates[-1],
    ))

    class RecordingSequenceDetector:
        model_fingerprint = "unit-model"

        def __init__(self):
            self.calls = []

        def detect_sequence(self, images, initial_context=None):
            self.calls.append((len(images), initial_context is not None))
            contexts = []
            for image_index in range(len(images)):
                value = float(image_index + len(self.calls))
                contexts.append({
                    "state": np.full((1, 1, 1, 1), value, dtype=np.float32),
                    "conv": np.full((1, 1, 1, 1), value, dtype=np.float32),
                    "spatial": np.full((1, 1, 1), value, dtype=np.float32),
                    "temporal": np.full((1, 1, 1), value, dtype=np.float32),
                })
            return ChangeDetectionOutput(
                mask=np.zeros(images[-1].shape[:2], dtype=np.uint8),
                confidence_score=0.4,
                changed_pixel_fraction=0.0,
                detector_name="mamba_temporal_ssm",
                metadata={"temporal_contexts": contexts},
            )

    detector = RecordingSequenceDetector()
    state_store = TemporalStateStore(str(tmp_path / "states.h5"))
    service = ChangeDetectionService(
        detector,
        image_service,
        comparison_service,
        SummaryService(),
        temporal_state_store=state_store,
    )
    request = ChangeDetectionRequest(comparison_id=comparison.comparison_id)

    assert service.run_detection(request).status == "completed"
    assert detector.calls[-1] == (3, False)
    assert all(state_store.load(tile_id, date, detector.model_fingerprint) for date in dates)

    assert service.run_detection(request).status == "completed"
    assert detector.calls[-1] == (2, True)


def test_mamba_detector_is_selectable_through_the_api_and_reuses_state(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import config
    from main import create_app

    config_values = {
        "TERRAEYES_EMBEDDING_MODEL": "mock",
        "TERRAEYES_CHANGE_DETECTOR": "mamba_cd",
        "TERRAEYES_EMBEDDING_DIM": 512,
        "TERRAEYES_DB_PATH": str(tmp_path / "api.db"),
        "TERRAEYES_FAISS_INDEX_PATH": str(tmp_path / "api.index"),
        "TERRAEYES_TEMPORAL_STORE_PATH": str(tmp_path / "temporal.h5"),
        "TERRAEYES_DATA_DIR": str(tmp_path / "data"),
        "TERRAEYES_MODELS_DIR": str(tmp_path / "models"),
        "TERRAEYES_OSCD_DIR": str(Path(__file__).resolve().parents[2] / "images"),
        "TERRAEYES_MIN_CHANGE_AREA_PX": 1,
    }
    for name, value in config_values.items():
        monkeypatch.setattr(config.settings, name, value)

    checkpoint_path = tmp_path / "mamba.pt"
    model_config = {"in_channels": 3, "feature_dim": 16, "state_dim": 4}
    model = MambaTemporalChangeNet(**model_config)
    torch.save({
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "threshold": 0.5,
    }, checkpoint_path)
    monkeypatch.setattr(config.settings, "TERRAEYES_MAMBA_CHECKPOINT_PATH", str(checkpoint_path))

    payload = {
        "location_id": "paris",
        "tile_id": "paris_0001_0001_sentinel-2",
        "date_before": "2016-03-15",
        "date_after": "2018-06-20",
    }
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health").json()
        first = client.post("/api/v1/change-detection", json=payload)
        second = client.post("/api/v1/change-detection", json=payload)

    assert health["change_detector"] == "mamba_cd"
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["summary"]["detector"] == "mamba_temporal_ssm"
    assert (tmp_path / "temporal.h5").is_file()
