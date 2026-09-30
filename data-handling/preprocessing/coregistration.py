"""Temporal co-registration algorithms (ECC / Phase Correlation)."""
from typing import Tuple
import numpy as np


def phase_correlation_offset(image1: np.ndarray, image2: np.ndarray) -> Tuple[float, float]:
    """
    Computes sub-pixel translation (dy, dx) between two images using 2D phase correlation.
    """
    im1 = np.mean(image1, axis=-1) if image1.ndim == 3 else image1
    im2 = np.mean(image2, axis=-1) if image2.ndim == 3 else image2

    f1 = np.fft.fft2(im1)
    f2 = np.fft.fft2(im2)

    cross_power = (f2 * np.conj(f1)) / (np.abs(f2 * np.conj(f1)) + 1e-9)
    corr = np.fft.ifft2(cross_power)
    corr = np.fft.fftshift(np.real(corr))

    h, w = im1.shape
    y_peak, x_peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy = float(y_peak - h // 2)
    dx = float(x_peak - w // 2)

    return dy, dx


def co_register_ecc(
    reference: np.ndarray,
    target: np.ndarray,
    max_offset_px: int = 15
) -> Tuple[np.ndarray, float, float]:
    """
    Co-registers target image to reference image.
    Returns (aligned_target, dy, dx).
    """
    dy, dx = phase_correlation_offset(reference, target)

    # Check bounds
    if abs(dy) > max_offset_px or abs(dx) > max_offset_px:
        # Offset too large to safely shift without losing content
        return target, 0.0, 0.0

    shift_y = int(round(dy))
    shift_x = int(round(dx))

    aligned = np.roll(target, shift=(-shift_y, -shift_x), axis=(0, 1))
    return aligned, dy, dx
