"""Confidence score calculation and calibration utilities."""
from app.schemas.common import Confidence


def compute_confidence(
    changed_pixel_fraction: float,
    detector_score: float,
    calibrated: bool = False
) -> Confidence:
    """
    Computes blended confidence score combining detector signal and spatial consistency.
    """
    # Balance raw score and realistic coverage
    score = 0.7 * detector_score + 0.3 * min(changed_pixel_fraction * 5.0, 1.0)
    score = max(0.1, min(score, 0.99))

    return Confidence(
        score=round(score, 2),
        method="pixel_fraction_and_detector_blended",
        calibrated=calibrated
    )
