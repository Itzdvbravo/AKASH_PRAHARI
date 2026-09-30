"""Alignment verification between bi-temporal tile pairs."""
from typing import Tuple
import numpy as np


def verify_tile_alignment(
    tile_before: np.ndarray,
    tile_after: np.ndarray,
    max_divergence_threshold: float = 0.8
) -> Tuple[bool, float]:
    """
    Checks if two tiles are pixel-aligned and comparable.
    Returns (is_aligned, structural_similarity_proxy).
    """
    if tile_before.shape != tile_after.shape:
        return False, 0.0

    b1 = np.mean(tile_before, axis=-1) if tile_before.ndim == 3 else tile_before
    b2 = np.mean(tile_after, axis=-1) if tile_after.ndim == 3 else tile_after

    # Compute correlation coefficient
    std1 = np.std(b1)
    std2 = np.std(b2)
    if std1 < 1e-4 or std2 < 1e-4:
        return True, 1.0

    corr = np.mean((b1 - np.mean(b1)) * (b2 - np.mean(b2))) / (std1 * std2)
    # A negative or highly diverged correlation might indicate complete sensor mismatch or misalignment
    is_aligned = bool(corr > -0.5)
    return is_aligned, float(corr)
