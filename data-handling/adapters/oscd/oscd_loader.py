"""OSCD dataset loader for discovering and reading Sentinel-2 multi-band TIFFs."""
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class OSCDLoader:
    """Discovers and loads Sentinel-2 band imagery and change masks from OSCD."""

    RGB_BANDS = ["B04", "B03", "B02"]  # Red, Green, Blue
    INFRARED_BANDS = ["B08", "B04", "B03"]  # NIR, Red, Green

    def __init__(self, oscd_root_dir: Path):
        self.oscd_root = Path(oscd_root_dir)

    def find_city_directories(self) -> Dict[str, Path]:
        """Search recursively for city folders in OSCD dataset."""
        cities = {}
        if not self.oscd_root.exists():
            return cities

        # Look in oscd_root or subdirectories like "Onera Satellite Change Detection dataset - Images"
        candidates = list(self.oscd_root.glob("**/imgs_1*"))
        if not candidates:
            # Maybe directories are directly under root
            for item in self.oscd_root.iterdir():
                if item.is_dir():
                    if (item / "imgs_1_rect").exists() or (item / "imgs_1").exists():
                        cities[item.name.lower()] = item
        else:
            for c in candidates:
                city_dir = c.parent
                cities[city_dir.name.lower()] = city_dir

        return cities

    def _read_single_band(self, band_file: Path) -> np.ndarray:
        """Reads a single band file (.tif, .png, etc.) returning float32 array normalized to [0, 1]."""
        if HAS_RASTERIO:
            try:
                with rasterio.open(band_file) as src:
                    arr = src.read(1).astype(np.float32)
                    # Normalize Sentinel-2 surface reflectance (typical max ~10,000 DN)
                    p98 = np.percentile(arr, 98)
                    norm_denom = max(p98, 4000.0)
                    arr = np.clip(arr / norm_denom, 0.0, 1.0)
                    return arr
            except Exception:
                pass

        # Fallback to PIL
        with Image.open(band_file) as img:
            arr = np.array(img, dtype=np.float32)
            if arr.max() > 1.0:
                max_val = np.percentile(arr, 98) if arr.max() > 255 else 255.0
                max_val = max(max_val, 1.0)
                arr = np.clip(arr / max_val, 0.0, 1.0)
            return arr

    def load_band_composite(
        self, city_dir: Path, time_index: int = 1, bands: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Load composite image (H, W, C) from imgs_{time_index}_rect or imgs_{time_index}.
        time_index: 1 for first acquisition, 2 for second acquisition.
        """
        if bands is None:
            bands = self.RGB_BANDS

        # Determine imgs dir
        img_dir = city_dir / f"imgs_{time_index}_rect"
        if not img_dir.exists():
            img_dir = city_dir / f"imgs_{time_index}"
        if not img_dir.exists():
            # Try subfolder matching time_index
            subdirs = sorted([d for d in city_dir.iterdir() if d.is_dir()])
            if len(subdirs) >= time_index:
                img_dir = subdirs[time_index - 1]
            else:
                raise FileNotFoundError(f"Could not locate image directory for time index {time_index} in {city_dir}")

        channel_arrays = []
        for band in bands:
            band_candidates = list(img_dir.glob(f"*{band}*.tif")) + list(img_dir.glob(f"*{band}*.tiff"))
            if not band_candidates:
                # Search case-insensitively
                band_candidates = [
                    f for f in img_dir.iterdir() if f.is_file() and band.lower() in f.name.lower()
                ]

            if band_candidates:
                channel_arrays.append(self._read_single_band(band_candidates[0]))
            else:
                # If specific band not found and an RGB tif/png exists, load it directly
                rgb_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
                if rgb_files:
                    with Image.open(rgb_files[0]) as img:
                        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                        return arr
                raise FileNotFoundError(f"Band {band} not found in {img_dir}")

        # Stack into (H, W, C)
        composite = np.stack(channel_arrays, axis=-1)
        return composite.astype(np.float32)

    def load_ground_truth_mask(self, city_name: str) -> Optional[np.ndarray]:
        """Find and load binary ground-truth change mask for a city."""
        # Check standard locations: Train Labels/<city>/cm/cm.png or <city>/cm/cm.png
        patterns = [
            f"**/{city_name}/cm/cm.png",
            f"**/{city_name}/cm/*.png",
            f"**/{city_name}/*cm*.tif",
            f"**/{city_name}/*mask*.png",
        ]

        for pat in patterns:
            matches = list(self.oscd_root.glob(pat))
            if matches:
                mask_file = matches[0]
                with Image.open(mask_file) as img:
                    mask = np.array(img.convert("L"))
                    # In OSCD, 1 or 255 signifies change, 2 or 0 signifies no change
                    binary_mask = (mask == 255) | (mask == 1)
                    return binary_mask.astype(np.uint8)

        return None
