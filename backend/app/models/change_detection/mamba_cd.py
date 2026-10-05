"""Checkpoint-backed multi-temporal Mamba selective-SSM detector."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from app.models.interfaces import ChangeDetectionOutput, ChangeDetector
from .mamba_model import MambaTemporalChangeNet


class MambaChangeDetector(ChangeDetector):
    """Run a trained temporal SSM checkpoint on a sequence of aligned images."""

    def __init__(
        self,
        weights_path: str,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
    ):
        import torch

        self.weights_path = Path(weights_path)
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Mamba checkpoint not found: {self.weights_path}. "
                "Train it with scripts/train_mamba_cd.py before enabling mamba_cd."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # This checkpoint is generated locally by our training script and stores
        # only primitive metadata plus a tensor state dict.
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("model_config", {})
        self.model = MambaTemporalChangeNet(**config)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.threshold = float(threshold if threshold is not None else checkpoint["threshold"])
        self.model_fingerprint = hashlib.sha256(self.weights_path.read_bytes()).hexdigest()

    def detect(self, before: np.ndarray, after: np.ndarray) -> ChangeDetectionOutput:
        return self.detect_sequence([before, after])

    def detect_sequence(
        self,
        images: Sequence[np.ndarray],
        initial_context: Optional[dict[str, np.ndarray]] = None,
    ) -> ChangeDetectionOutput:
        import torch

        if len(images) < 1:
            raise ValueError("At least one image is required for temporal inference")
        normalized: list[np.ndarray] = []
        expected_shape: Optional[tuple[int, ...]] = None
        for image in images:
            array = np.asarray(image, dtype=np.float32)
            if array.ndim != 3 or array.shape[-1] != self.model.in_channels:
                raise ValueError(
                    f"Expected HxWx{self.model.in_channels} image arrays; got {array.shape}"
                )
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise ValueError("All dates in a temporal sequence must have matching image shapes")
            if array.size and float(np.nanmax(array)) > 1.0:
                array = array / 255.0
            normalized.append(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0))

        image_tensor = torch.from_numpy(np.stack(normalized)).permute(0, 3, 1, 2).unsqueeze(0)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)
        model_args: dict[str, Any] = {}
        if initial_context:
            model_args["initial_state"] = torch.from_numpy(initial_context["state"]).unsqueeze(0)
            model_args["initial_conv_state"] = torch.from_numpy(initial_context["conv"]).unsqueeze(0)
            model_args["previous_spatial"] = torch.from_numpy(initial_context["spatial"]).unsqueeze(0)
            model_args["previous_temporal"] = torch.from_numpy(initial_context["temporal"]).unsqueeze(0)
        model_args = {
            name: tensor.to(device=self.device, dtype=torch.float32)
            for name, tensor in model_args.items()
        }

        with torch.inference_mode():
            logits, state_snapshots, conv_snapshots, spatial_features, temporal_features = self.model(
                image_tensor,
                return_context=True,
                **model_args,
            )
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask = (probabilities >= self.threshold).astype(np.uint8)
        changed_fraction = float(mask.mean())
        confidence = float(probabilities[mask > 0].mean()) if np.any(mask) else float(probabilities.max())
        contexts = [
            {
                "state": state[0].detach().cpu().numpy(),
                "conv": conv[0].detach().cpu().numpy(),
                "spatial": spatial[0].detach().cpu().numpy(),
                "temporal": temporal[0].detach().cpu().numpy(),
            }
            for state, conv, spatial, temporal in zip(
                state_snapshots, conv_snapshots, spatial_features, temporal_features
            )
        ]
        return ChangeDetectionOutput(
            mask=mask,
            confidence_score=round(float(np.clip(confidence, 0.0, 1.0)), 4),
            changed_pixel_fraction=round(changed_fraction, 4),
            detector_name="mamba_temporal_ssm",
            metadata={
                "threshold": self.threshold,
                "temporal_length": len(images) + (1 if initial_context else 0),
                "temporal_contexts": contexts,
                "device": str(self.device),
            },
        )
