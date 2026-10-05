"""Fixed-threshold pixel-difference detector for DynamicEarthNet RGB tiles."""
import numpy as np
from app.models.interfaces import ChangeDetector, ChangeDetectionOutput

class PixelDiffChangeDetector(ChangeDetector):
    """
    Non-parametric binary baseline using max-channel PlanetFusion RGB difference.
    """

    def __init__(self, threshold: float = 0.3):
        self.threshold = float(np.clip(threshold, 0.01, 0.95))

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

        # Otsu and the optical cloud heuristic both performed poorly on
        # PlanetFusion RGB: Otsu over-thresholded sparse changes, while the
        # cloud heuristic erased genuine bright land-cover changes. Use the
        # fixed threshold selected from a DynamicEarthNet validation sample.
        binary_mask = (diff_map >= self.threshold).astype(np.uint8)

        total_px = float(binary_mask.size)
        changed_px = float(np.count_nonzero(binary_mask))
        changed_fraction = changed_px / max(total_px, 1.0)

        # Confidence heuristic: higher if change signal is distinct from noise
        mean_change_diff = float(np.mean(diff_map[binary_mask == 1])) if changed_px > 0 else 0.0
        confidence_score = (
            float(np.clip(mean_change_diff * (1.0 - 0.5 * changed_fraction), 0.0, 0.98))
            if changed_px else 0.0
        )

        return ChangeDetectionOutput(
            mask=binary_mask,
            confidence_score=round(confidence_score, 4),
            changed_pixel_fraction=round(changed_fraction, 4),
            detector_name="pixel_diff_fixed_threshold",
            metadata={
                "threshold_applied": round(self.threshold, 4),
                "diff_map_mean": round(float(np.mean(diff_map)), 4),
                "diff_map_max": round(float(np.max(diff_map)), 4),
                "cloud_filter": "disabled; it suppressed genuine PlanetFusion changes",
            }
        )
