"""Vector store abstraction interface."""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class VectorStore(ABC):
    """Abstract Base Class for offline Vector Index implementations."""

    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: List[str]) -> None:
        """Add normalized embeddings with corresponding tile IDs."""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search top-k nearest neighbours, returning list of (id, cosine_similarity_score)."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist index to disk."""
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load index from disk."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Total vectors currently in index."""
        pass
