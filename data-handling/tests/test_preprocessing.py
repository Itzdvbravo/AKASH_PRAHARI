"""Unit tests for preprocessing: normalization, cloud masking, tiling, coregistration."""
import numpy as np
from preprocessing.normalization import normalize_bands, z_score_normalize
from preprocessing.cloud_masking import detect_clouds_optical
from preprocessing.tiling import tile_scene, assemble_tiles, generate_tile_id
from preprocessing.coregistration import phase_correlation_offset, co_register_ecc
from preprocessing.pipeline import PreprocessingPipeline


def test_normalize_bands(sample_scene_array):
    raw = sample_scene_array * 4000.0  # Simulated DN values
    norm = normalize_bands(raw)
    assert norm.shape == sample_scene_array.shape
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_z_score_normalize(sample_scene_array):
    standardized, mean, std = z_score_normalize(sample_scene_array)
    assert standardized.shape == sample_scene_array.shape
    assert np.isclose(np.mean(standardized), 0.0, atol=1e-1)


def test_cloud_masking():
    # Construct an image with a bright white cloud in the center
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[:, :] = [0.2, 0.4, 0.2]  # Vegetation
    img[30:70, 30:70] = [0.95, 0.95, 0.95]  # Bright cloud

    mask = detect_clouds_optical(img, whiteness_threshold=0.8, hot_threshold=0.7)
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 1  # Cloud detected
    assert mask[10, 10] == 0  # Clear ground


def test_tiling_and_assembly(sample_scene_array):
    tiles = tile_scene(sample_scene_array, tile_size=256, stride=256, location_id="paris")
    assert len(tiles) == 4  # 512x512 with 256 tiles -> 2x2 grid = 4 tiles

    tile_id, tile_arr, coords = tiles[0]
    assert tile_arr.shape == (256, 256, 3)
    assert "paris_0000_0000" in tile_id

    # Assemble back
    tile_tuples = [(t[1], t[2]) for t in tiles]
    reconstructed = assemble_tiles(tile_tuples, (512, 512))
    assert reconstructed.shape == sample_scene_array.shape


def test_coregistration():
    ref = np.zeros((128, 128), dtype=np.float32)
    ref[40:80, 40:80] = 1.0

    # Shifted by (dy=3, dx=4)
    target = np.roll(ref, shift=(3, 4), axis=(0, 1))

    dy, dx = phase_correlation_offset(ref, target)
    assert round(dy) == 3
    assert round(dx) == 4


def test_preprocessing_pipeline(sample_scene_array):
    pipeline = PreprocessingPipeline(sensor="sentinel-2")
    record = pipeline.run(sample_scene_array)

    assert "image" in record
    assert "cloud_mask" in record
    assert record["image"].shape == sample_scene_array.shape
