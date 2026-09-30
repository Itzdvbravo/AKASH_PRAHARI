"""Cloud and shadow masking for optical satellite sensors (Sentinel-2, Landsat)."""
from typing import Optional
import numpy as np


def detect_clouds_optical(
    rgb_nir_image: np.ndarray,
    whiteness_threshold: float = 0.7,
    hot_threshold: float = 0.6
) -> np.ndarray:
    """
    Computes an optical cloud mask from normalized RGB/NIR bands.
    rgb_nir_image: shape (H, W, C) where C >= 3 (R, G, B) or (R, G, B, NIR).
    Returns binary mask (H, W) where 1 indicates cloud/haze and 0 indicates clear.
    """
    r = rgb_nir_image[..., 0]
    g = rgb_nir_image[..., 1]
    b = rgb_nir_image[..., 2]

    # Whiteness index: clouds reflect evenly across visible spectrum
    mean_vis = (r + g + b) / 3.0
    whiteness = 1.0 - (np.abs(r - mean_vis) + np.abs(g - mean_vis) + np.abs(b - mean_vis)) / (3.0 * (mean_vis + 1e-6))

    # Brightness index: clouds have high reflectance in visible spectrum
    brightness = mean_vis

    # Combined cloud criterion
    cloud_mask = (brightness > hot_threshold) & (whiteness > whiteness_threshold)
    return cloud_mask.astype(np.uint8)
