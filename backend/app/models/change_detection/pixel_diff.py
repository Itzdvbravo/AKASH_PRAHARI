"""Baseline pixel-difference and Otsu threshold change detector."""
import numpy as np
from app.models.interfaces import ChangeDetector, ChangeDetectionOutput

try:
    from scipy.ndimage import binary_dilation, generate_binary_structure
    HAS_SCIPY_MORPH = True
except ImportError:
    HAS_SCIPY_MORPH = False

import sys
from pathlib import Path
_dh_root = Path(__file__).resolve().parents[4] / "data-handling"
if str(_dh_root) not in sys.path:
    sys.path.insert(0, str(_dh_root))
from preprocessing.cloud_masking import detect_clouds_optical


try:
    from skimage.filters import threshold_otsu
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class PixelDiffChangeDetector(ChangeDetector):
    """
    Non-parametric baseline change detector using spectral pixel difference and Otsu thresholding.
    """

    def __init__(self, min_threshold: float = 0.15, max_threshold: float = 0.85):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def detect(self, before: np.ndarray, after: np.ndarray) -> ChangeDetectionOutput:
        # Validate shapes
        if before.shape != after.shape:
            raise ValueError(f"Shape mismatch in change detector: before {before.shape} vs after {after.shape}")

        # Ensure float32 in [0, 1]
        b = before.astype(np.float32)
        a = after.astype(np.float32)
        if b.max() > 1.0:
            b /= 255.0
        if a.max() > 1.0:
            a /= 255.0

        # Absolute per-channel difference
        diff = np.abs(a - b)

        # Single-channel difference map: maximum difference across spectral channels
        if diff.ndim == 3:
            diff_map = np.max(diff, axis=-1)
        else:
            diff_map = diff

        # Otsu thresholding
        thresh = self.min_threshold
        if HAS_SKIMAGE and diff_map.max() > diff_map.min():
            try:
                otsu_val = float(threshold_otsu(diff_map))
                thresh = float(np.clip(otsu_val, self.min_threshold, self.max_threshold))
            except Exception:
                thresh = self.min_threshold
        else:
            # Fallback simple median + std
            thresh = float(np.clip(np.mean(diff_map) + np.std(diff_map), self.min_threshold, self.max_threshold))

        binary_mask = (diff_map >= thresh).astype(np.uint8)

        # --- Cloud suppression ---
        # Detect clouds in both images and exclude those pixels from the
        # change mask to avoid false positives from transient atmospheric
        # features (clouds, haze, shadows).
        cloud_before = detect_clouds_optical(b) if b.ndim == 3 and b.shape[-1] >= 3 else np.zeros_like(binary_mask)
        cloud_after = detect_clouds_optical(a) if a.ndim == 3 and a.shape[-1] >= 3 else np.zeros_like(binary_mask)
        cloud_union = ((cloud_before > 0) | (cloud_after > 0)).astype(np.uint8)

        # Dilate cloud mask slightly to catch edges / thin haze around clouds
        if HAS_SCIPY_MORPH and np.any(cloud_union):
            struct = generate_binary_structure(2, 2)
            cloud_union = binary_dilation(cloud_union, structure=struct, iterations=3).astype(np.uint8)

        binary_mask = binary_mask & (~cloud_union.astype(bool)).astype(np.uint8)


        total_px = float(binary_mask.size)
        changed_px = float(np.count_nonzero(binary_mask))
        changed_fraction = changed_px / max(total_px, 1.0)

        # Confidence heuristic: higher if change signal is distinct from noise
        mean_change_diff = float(np.mean(diff_map[binary_mask == 1])) if changed_px > 0 else 0.0
        confidence_score = float(np.clip(mean_change_diff * (1.0 - 0.5 * changed_fraction), 0.1, 0.98))

        return ChangeDetectionOutput(
            mask=binary_mask,
            confidence_score=round(confidence_score, 4),
            changed_pixel_fraction=round(changed_fraction, 4),
            detector_name="pixel_diff_otsu",
            metadata={
                "threshold_applied": round(thresh, 4),
                "diff_map_mean": round(float(np.mean(diff_map)), 4),
                "diff_map_max": round(float(np.max(diff_map)), 4),
                "cloud_pixels_suppressed": int(np.count_nonzero(cloud_union)),
            }
        )
