"""Band normalization and reflectance conversion utilities."""
from typing import Optional, Tuple
import numpy as np


def normalize_bands(
    image: np.ndarray,
    min_percentile: float = 2.0,
    max_percentile: float = 98.0,
    clip: bool = True
) -> np.ndarray:
    """
    Normalizes multi-band image (H, W, C) to [0.0, 1.0] using per-channel percentiles.
    """
    normalized = np.zeros_like(image, dtype=np.float32)
    channels = image.shape[-1] if image.ndim == 3 else 1

    for c in range(channels):
        channel_data = image[..., c] if image.ndim == 3 else image
        p_min = np.percentile(channel_data, min_percentile)
        p_max = np.percentile(channel_data, max_percentile)
        denom = max(p_max - p_min, 1e-6)

        norm_c = (channel_data - p_min) / denom
        if clip:
            norm_c = np.clip(norm_c, 0.0, 1.0)

        if image.ndim == 3:
            normalized[..., c] = norm_c
        else:
            normalized = norm_c

    return normalized


def z_score_normalize(
    image: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel z-score standardization (zero mean, unit variance).
    Returns (standardized_image, mean, std).
    """
    if mean is None:
        mean = np.mean(image, axis=(0, 1), keepdims=True)
    if std is None:
        std = np.std(image, axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1.0, std)

    standardized = (image - mean) / std
    return standardized.astype(np.float32), mean, std
