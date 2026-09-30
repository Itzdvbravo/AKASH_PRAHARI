"""Speckle noise filtering for SAR imagery (e.g. Sentinel-1)."""
import numpy as np


def lee_filter(image: np.ndarray, window_size: int = 5, damping_factor: float = 1.0) -> np.ndarray:
    """
    Lee filter for speckle reduction in SAR intensity or amplitude images.
    image: (H, W) or (H, W, C).
    """
    from scipy.ndimage import uniform_filter

    is_3d = image.ndim == 3
    channels = image.shape[-1] if is_3d else 1
    filtered = np.zeros_like(image, dtype=np.float32)

    for c in range(channels):
        band = image[..., c] if is_3d else image
        mean = uniform_filter(band, (window_size, window_size))
        sq_mean = uniform_filter(band ** 2, (window_size, window_size))
        variance = np.maximum(sq_mean - mean ** 2, 0.0)

        overall_var = np.var(band)
        weight = variance / (variance + overall_var + 1e-6)
        weight = np.clip(weight * damping_factor, 0.0, 1.0)

        out = mean + weight * (band - mean)
        if is_3d:
            filtered[..., c] = out
        else:
            filtered = out

    return filtered
