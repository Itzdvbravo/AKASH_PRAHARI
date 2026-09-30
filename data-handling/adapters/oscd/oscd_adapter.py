"""OSCD DatasetAdapter implementation conforming to DatasetAdapter interface."""
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

import sys
# Ensure data-handling root is accessible for imports
data_handling_dir = Path(__file__).resolve().parents[2]
if str(data_handling_dir) not in sys.path:
    sys.path.insert(0, str(data_handling_dir))

from interfaces.dataset_adapter import DatasetAdapter, TileMetadata
from adapters.oscd.oscd_loader import OSCDLoader
from adapters.oscd.oscd_metadata import OSCDMetadataExtractor, OSCD_CITY_COORDINATES


class OSCDAdapter(DatasetAdapter):
    """Production DatasetAdapter for the Onera Satellite Change Detection (OSCD) dataset."""

    def __init__(self, oscd_root_dir: str):
        self.oscd_root = Path(oscd_root_dir)
        self.loader = OSCDLoader(self.oscd_root)
        self.metadata_extractor = OSCDMetadataExtractor(self.oscd_root)
        self._cities_cache: Optional[Dict[str, Path]] = None

    def _get_cities(self) -> Dict[str, Path]:
        if self._cities_cache is None:
            self._cities_cache = self.loader.find_city_directories()
        return self._cities_cache

    def list_locations(self) -> List[str]:
        cities = self._get_cities()
        if cities:
            return sorted(list(cities.keys()))
        # Fallback to predefined cities if directory not yet mounted
        return sorted(list(OSCD_CITY_COORDINATES.keys()))

    def list_dates(self, location_id: str) -> List[str]:
        cities = self._get_cities()
        city_dir = cities.get(location_id.lower())
        if city_dir:
            return self.metadata_extractor.get_dates_for_city(city_dir)
        return ["2016-01-01", "2018-01-01"]

    def get_tile_metadata(self, tile_id: str) -> TileMetadata:
        # Expected format: {location_id}_{row:04d}_{col:04d}_{sensor}
        parts = tile_id.split("_")
        loc_id = parts[0] if parts else "unknown"
        sensor = parts[-1] if len(parts) > 1 else "sentinel-2"

        geo_bbox = self.metadata_extractor.get_geo_bbox(loc_id)
        dates = self.list_dates(loc_id)
        default_date = dates[0] if dates else "2016-01-01"

        return TileMetadata(
            tile_id=tile_id,
            location_id=loc_id,
            sensor=sensor,
            date=default_date,
            width=256,
            height=256,
            channels=3,
            crs="EPSG:4326",
            geo_bbox=geo_bbox,
            additional_metadata={
                "dataset": "OSCD",
                "city": loc_id.capitalize(),
                "sensor_type": "optical",
            }
        )

    def load_tile_array(
        self, location_id: str, date: str, tile_id: str, bands: Optional[List[str]] = None
    ) -> np.ndarray:
        cities = self._get_cities()
        city_dir = cities.get(location_id.lower())
        if not city_dir or not city_dir.exists():
            raise FileNotFoundError(f"OSCD city directory not found for location: {location_id}")

        dates = self.list_dates(location_id)
        # Determine whether this is date 1 or date 2
        time_index = 1
        if len(dates) >= 2 and date == dates[1]:
            time_index = 2

        full_scene = self.loader.load_band_composite(city_dir, time_index=time_index, bands=bands)

        # Handle tiling if tile_id specifies row/col
        parts = tile_id.split("_")
        if len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit():
            row = int(parts[1])
            col = int(parts[2])
            tile_size = 256
            r_start = row * tile_size
            c_start = col * tile_size
            h, w = full_scene.shape[:2]
            r_end = min(r_start + tile_size, h)
            c_end = min(c_start + tile_size, w)

            tile = np.zeros((tile_size, tile_size, full_scene.shape[2]), dtype=np.float32)
            actual = full_scene[r_start:r_end, c_start:c_end, :]
            tile[0:(r_end - r_start), 0:(c_end - c_start), :] = actual
            return tile

        # Return full scene or center crop if no tile coordinates in ID
        h, w = full_scene.shape[:2]
        if h > 256 or w > 256:
            r_mid = h // 2
            c_mid = w // 2
            return full_scene[r_mid - 128:r_mid + 128, c_mid - 128:c_mid + 128, :]

        return full_scene

    def load_ground_truth_mask(
        self, location_id: str, date_before: str, date_after: str
    ) -> Optional[np.ndarray]:
        return self.loader.load_ground_truth_mask(location_id.lower())
