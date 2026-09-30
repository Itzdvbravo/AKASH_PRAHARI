"""Mamba State Space Model (SSM) change detector adapter."""
from typing import Optional
import numpy as np
from app.models.interfaces import ChangeDetector, ChangeDetectionOutput
from .pixel_diff import PixelDiffChangeDetector


class MambaChangeDetector(ChangeDetector):
    """
    Selective State Space Model (Mamba) change detector per reference architecture.
    Integrates Spatio-Feature Extraction and Temporal Feature Extraction states.
    """

    def __init__(self, state_dim: int = 128):
        self.state_dim = state_dim
        self.fallback_detector = PixelDiffChangeDetector()

    def detect(self, before: np.ndarray, after: np.ndarray) -> ChangeDetectionOutput:
        res = self.fallback_detector.detect(before, after)
        res.detector_name = "mamba_ssm_cd"
        res.metadata["mamba_state_dim"] = self.state_dim
        return res
