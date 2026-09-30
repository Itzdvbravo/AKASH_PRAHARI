"""Composable preprocessing pipeline for multi-sensor satellite imagery."""
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from .normalization import normalize_bands
from .cloud_masking import detect_clouds_optical
from .coregistration import co_register_ecc


class PreprocessingStep:
    """A single configurable step in the preprocessing pipeline."""

    def __init__(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]], enabled: bool = True):
        self.name = name
        self.fn = fn
        self.enabled = enabled

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return record
        return self.fn(record)


class PreprocessingPipeline:
    """Composable pipeline executing preprocessing steps sequentially."""

    def __init__(self, sensor: str = "sentinel-2"):
        self.sensor = sensor.lower()
        self.steps: List[PreprocessingStep] = []
        self._configure_default_steps()

    def _configure_default_steps(self) -> None:
        if self.sensor in ["sentinel-2", "landsat"]:
            # Step 1: Normalization
            def step_norm(rec: Dict[str, Any]) -> Dict[str, Any]:
                rec["image"] = normalize_bands(rec["image"])
                return rec

            # Step 2: Cloud detection
            def step_clouds(rec: Dict[str, Any]) -> Dict[str, Any]:
                if rec["image"].shape[-1] >= 3:
                    rec["cloud_mask"] = detect_clouds_optical(rec["image"])
                return rec

            # Step 3: Co-registration if reference image provided
            def step_coreg(rec: Dict[str, Any]) -> Dict[str, Any]:
                if "reference_image" in rec and rec["reference_image"] is not None:
                    aligned, dy, dx = co_register_ecc(rec["reference_image"], rec["image"])
                    rec["image"] = aligned
                    rec["coregistration_offset"] = (dy, dx)
                return rec

            self.steps.extend([
                PreprocessingStep("normalize", step_norm),
                PreprocessingStep("cloud_mask", step_clouds),
                PreprocessingStep("coregister", step_coreg),
            ])

    def add_step(self, step: PreprocessingStep) -> None:
        self.steps.append(step)

    def run(self, image: np.ndarray, reference_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "image": image,
            "reference_image": reference_image,
            "sensor": self.sensor,
            "metadata": {}
        }
        for step in self.steps:
            record = step(record)
        return record
