"""Embedding models package."""
from .mock_embedding import MockEmbeddingModel
from .remote_clip import RemoteCLIPEmbeddingModel

__all__ = ["MockEmbeddingModel", "RemoteCLIPEmbeddingModel"]
