from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from .dataset_adapter import TileMetadata


class TileProvider(ABC):
    """Abstract Base Class for tile rendering and streaming services."""

    @abstractmethod
    def get_tile_bytes(
        self, tile_id: str, date: str, fmt: str = "png", bands: Optional[List[str]] = None
    ) -> bytes:
        """Render tile pixels to raw image format bytes (PNG/JPEG)."""
        pass

    @abstractmethod
    def get_tile_metadata(self, tile_id: str) -> TileMetadata:
        """Fetch metadata for tile."""
        pass
