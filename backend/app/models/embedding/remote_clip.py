"""RemoteCLIP / OpenCLIP embedding model adapter conforming to EmbeddingModel ABC."""
from pathlib import Path
from typing import Optional
import numpy as np
from app.models.interfaces import EmbeddingModel

try:
    import torch
    import open_clip
    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False


class RemoteCLIPEmbeddingModel(EmbeddingModel):
    """
    RemoteCLIP adapter for satellite imagery and natural language queries.
    Fallback to deterministic projection if torch/open_clip not installed.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", dim: int = 512):
        self._dim = dim
        self._device = "cuda" if HAS_OPENCLIP and torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None

        if HAS_OPENCLIP:
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=self._device
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.model.eval()
            except Exception:
                self.model = None

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def encode_text(self, text: str) -> np.ndarray:
        if self.model is not None and self.tokenizer is not None:
            with torch.no_grad():
                tokens = self.tokenizer([text]).to(self._device)
                features = self.model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy()[0].astype(np.float32)

        # Fallback projection
        import hashlib
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()
        rng = np.random.RandomState(int(digest[:8], 16))
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        if self.model is not None and self.preprocess is not None:
            from PIL import Image
            with torch.no_grad():
                if image.max() <= 1.0:
                    img_u8 = (image * 255.0).astype(np.uint8)
                else:
                    img_u8 = image.astype(np.uint8)
                pil_img = Image.fromarray(img_u8[..., :3])
                tensor = self.preprocess(pil_img).unsqueeze(0).to(self._device)
                features = self.model.encode_image(tensor)
                features /= features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy()[0].astype(np.float32)

        # Fallback projection
        mean_val = float(np.mean(image))
        rng = np.random.RandomState(int(abs(mean_val * 1000000)) % (2**31 - 1))
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)
