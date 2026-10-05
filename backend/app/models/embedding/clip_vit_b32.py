"""Offline OpenAI CLIP ViT-B/32 embedding adapter."""
from pathlib import Path
from typing import Any

import numpy as np

from app.models.interfaces import EmbeddingModel


class CLIPViTB32EmbeddingModel(EmbeddingModel):
    """Loads CLIP only from a local OpenCLIP state-dict checkpoint."""

    def __init__(self, checkpoint_path: str, device: str | None = None):
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"CLIP ViT-B/32 checkpoint not found at {checkpoint}. "
                "Run scripts/download_models.py before starting the API."
            )

        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "CLIP ViT-B/32 requires torch and open_clip_torch. "
                "Install the project requirements before selecting this model."
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu",
            pretrained=str(checkpoint.resolve()),
            device=self._device,
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        self.model.eval()
        self._dim = int(self.model.text_projection.shape[-1])
        if self._dim != 512:
            raise ValueError(f"Expected CLIP ViT-B/32 dimension 512; got {self._dim}")

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _normalize(self, features: Any) -> np.ndarray:
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result = features.detach().float().cpu().numpy()[0].astype(np.float32)
        if result.shape != (self._dim,):
            raise ValueError(f"Unexpected CLIP embedding shape: {result.shape}")
        return result

    def encode_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a batch of RGB images with one model forward pass."""
        from PIL import Image

        if not images:
            return np.empty((0, self._dim), dtype=np.float32)
        tensors = []
        for image in images:
            pixels = np.asarray(image)
            if pixels.ndim == 2:
                pixels = np.repeat(pixels[..., None], 3, axis=-1)
            if pixels.ndim != 3 or pixels.shape[-1] < 3:
                raise ValueError("CLIP image input must have shape (height, width, >=3)")
            pixels = np.nan_to_num(pixels[..., :3], nan=0.0, posinf=1.0, neginf=0.0)
            if np.issubdtype(pixels.dtype, np.integer) and pixels.max(initial=0) > 255:
                upper = np.percentile(pixels, 98, axis=(0, 1), keepdims=True)
                pixels = pixels.astype(np.float32) / np.maximum(upper, 1.0) * 255.0
            elif pixels.max(initial=0) <= 1.0:
                pixels = pixels.astype(np.float32) * 255.0
            pil_image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")
            tensors.append(self.preprocess(pil_image))
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.inference_mode():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.detach().float().cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self._device)
        with self._torch.inference_mode():
            return self._normalize(self.model.encode_text(tokens))

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        pixels = np.asarray(image)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[..., None], 3, axis=-1)
        if pixels.ndim != 3 or pixels.shape[-1] < 3:
            raise ValueError("CLIP image input must have shape (height, width, >=3)")
        pixels = np.nan_to_num(pixels[..., :3], nan=0.0, posinf=1.0, neginf=0.0)
        if np.issubdtype(pixels.dtype, np.integer) and pixels.max(initial=0) > 255:
            # Handle raw 16-bit TIFF arrays if callers bypass the normalized adapter.
            upper = np.percentile(pixels, 98, axis=(0, 1), keepdims=True)
            pixels = pixels.astype(np.float32) / np.maximum(upper, 1.0) * 255.0
        elif pixels.max(initial=0) <= 1.0:
            pixels = pixels.astype(np.float32) * 255.0
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(pixels, mode="RGB")
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            return self._normalize(self.model.encode_image(tensor))
