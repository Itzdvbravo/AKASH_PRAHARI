from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class ChangeDetectionOutput:
    mask: np.ndarray  # 2D boolean or uint8 binary array (H, W) where 1/True represents change
    confidence_score: float  # [0.0, 1.0]
    changed_pixel_fraction: float
    detector_name: str
    metadata: Optional[Dict[str, Any]] = None


class EmbeddingModel(ABC):
    """Abstract Base Class for Vision-Language and Image Embedding Models."""

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query string into L2-normalized 1D float32 embedding vector."""
        pass

    @abstractmethod
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Encode image tile array (H, W, C) into L2-normalized 1D float32 embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of vector embedding output (e.g., 512, 768)."""
        pass


class ChangeDetector(ABC):
    """Abstract Base Class for Bi-Temporal Change Detection Algorithms and Models."""

    @abstractmethod
    def detect(self, before: np.ndarray, after: np.ndarray) -> ChangeDetectionOutput:
        """
        Detect changes between 'before' and 'after' image tile arrays of shape (H, W, C).
        Returns ChangeDetectionOutput containing binary mask and confidence metrics.
        """
        pass
