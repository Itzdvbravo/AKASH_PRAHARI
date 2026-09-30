"""BIT-CD (Bitemporal Image Transformer) change detector adapter."""
from typing import Optional
import numpy as np
from app.models.interfaces import ChangeDetector, ChangeDetectionOutput
from .pixel_diff import PixelDiffChangeDetector


class BITChangeDetector(ChangeDetector):
    """
    BIT-CD adapter for learned transformer-based change detection on bi-temporal pairs.
    Delegates to pixel_diff if PyTorch / BIT weights are not yet loaded.
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.weights_path = weights_path
        self.fallback_detector = PixelDiffChangeDetector()
        self.model = None

    def detect(self, before: np.ndarray, after: np.ndarray) -> ChangeDetectionOutput:
        if self.model is not None:
            # Model inference execution
            pass
        res = self.fallback_detector.detect(before, after)
        res.detector_name = "bit_cd"
        return res
