"""Morphological refinement of raw binary change detection masks."""
import numpy as np

try:
    from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def refine_change_mask(
    mask: np.ndarray,
    min_area_px: int = 16,
    close_radius: int = 2
) -> np.ndarray:
    """
    Cleans change mask:
    1. Removes isolated 1-pixel noise (opening).
    2. Fills pinholes within detected change areas (closing).
    """
    if not HAS_SCIPY:
        return mask.astype(np.uint8)

    structure = generate_binary_structure(2, 2)
    cleaned = binary_opening(mask > 0, structure=structure, iterations=1)
    if close_radius > 0:
        cleaned = binary_closing(cleaned, structure=structure, iterations=close_radius)

    return cleaned.astype(np.uint8)
