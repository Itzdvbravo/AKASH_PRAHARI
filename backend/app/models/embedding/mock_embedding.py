"""Mock embedding model producing deterministic unit vectors for tests and prototype."""
import hashlib
import numpy as np
from app.models.interfaces import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic mock embedding model conforming to EmbeddingModel ABC."""

    def __init__(self, dim: int = 512):
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _hash_to_vector(self, seed_int: int) -> np.ndarray:
        rng = np.random.RandomState(seed_int)
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode_text(self, text: str) -> np.ndarray:
        # Deterministic seed from text MD5
        digest = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        return self._hash_to_vector(seed)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        # Deterministic seed from downsampled image content
        down = image[::16, ::16]
        mean_val = float(np.mean(down))
        seed = int(abs(mean_val * 1000000)) % (2**31 - 1)
        return self._hash_to_vector(seed)
