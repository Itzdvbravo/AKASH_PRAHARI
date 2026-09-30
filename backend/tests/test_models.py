"""Unit tests for embedding and change detection models."""
import numpy as np
from app.models.embedding.mock_embedding import MockEmbeddingModel
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector


def test_mock_embedding_shape_and_norm():
    model = MockEmbeddingModel(dim=256)
    vec = model.encode_text("satellite imagery of rivers")
    assert vec.shape == (256,)
    assert vec.dtype == np.float32
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_mock_embedding_deterministic():
    model = MockEmbeddingModel(dim=256)
    v1 = model.encode_text("paris urban area")
    v2 = model.encode_text("paris urban area")
    assert np.allclose(v1, v2)

    v3 = model.encode_text("berlin forestry")
    assert not np.allclose(v1, v3)


def test_pixel_diff_change_detector(sample_tile_pair):
    detector = PixelDiffChangeDetector()
    before, after = sample_tile_pair
    result = detector.detect(before, after)

    assert result.mask.shape == (256, 256)
    assert result.mask.dtype in (np.uint8, bool)
    assert result.confidence_score > 0.0
    assert result.changed_pixel_fraction > 0.0
    # Region between (100, 100) and (150, 150) should be marked changed
    changed_roi = result.mask[100:150, 100:150]
    assert np.count_nonzero(changed_roi) > 0


def test_pixel_diff_identical_images():
    detector = PixelDiffChangeDetector()
    img = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    result = detector.detect(img, img)
    assert result.changed_pixel_fraction == 0.0
