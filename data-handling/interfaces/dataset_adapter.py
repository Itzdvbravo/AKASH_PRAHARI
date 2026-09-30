from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class TileMetadata:
    tile_id: str
    location_id: str
    sensor: str
    date: str
    width: int
    height: int
    channels: int
    crs: str
    geo_bbox: Dict[str, float]  # {"west": float, "south": float, "east": float, "north": float}
    additional_metadata: Optional[Dict[str, Any]] = None


class DatasetAdapter(ABC):
    """Abstract Base Class for Dataset Readers (e.g., OSCD, Sentinel-2, Landsat, Fixtures)."""

    @abstractmethod
    def list_locations(self) -> List[str]:
        """Return list of available location identifiers."""
        pass

    @abstractmethod
    def list_dates(self, location_id: str) -> List[str]:
        """Return available ISO 8601 acquisition dates for a given location."""
        pass

    @abstractmethod
    def get_tile_metadata(self, tile_id: str) -> TileMetadata:
        """Fetch metadata dataclass for a specific tile ID."""
        pass

    @abstractmethod
    def load_tile_array(
        self, location_id: str, date: str, tile_id: str, bands: Optional[List[str]] = None
    ) -> np.ndarray:
        """Load pixel data as float32 numpy array normalized to [0, 1] with shape (H, W, C)."""
        pass

    @abstractmethod
    def load_ground_truth_mask(
        self, location_id: str, date_before: str, date_after: str
    ) -> Optional[np.ndarray]:
        """Load ground truth binary change mask (H, W) if available, else None."""
        pass
