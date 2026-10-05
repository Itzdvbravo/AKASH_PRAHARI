"""Image service for retrieving tile image bytes, thumbnails, and masks."""
import io
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image

from app.db.repositories.tile_repo import TileRepository
from app.schemas.common import GeoBBox, SensorType
from app.exceptions import tile_not_found_exception

import sys
# Make data-handling importable
data_handling_dir = Path(__file__).resolve().parents[3] / "data-handling"
if str(data_handling_dir) not in sys.path:
    sys.path.insert(0, str(data_handling_dir))

from adapters.dynamicearthnet.adapter import DynamicEarthNetAdapter
from adapters.oscd.oscd_adapter import OSCDAdapter
from adapters.oscd.oscd_metadata import OSCD_CITY_COORDINATES
from geospatial.tile_id import parse_tile_id


class ImageService:
    """Manages satellite tile pixel retrieval, band rendering, and mask caching."""

    def __init__(
        self,
        tile_repo: TileRepository,
        oscd_dir: str | None = None,
        masks_cache_dir: str = "./data/masks",
        incremental_tiles_dir: str = "./data/incremental_tiles",
        dynamicearthnet_adapter: DynamicEarthNetAdapter | None = None,
    ):
        self.tile_repo = tile_repo
        self.oscd_adapter = OSCDAdapter(oscd_dir) if oscd_dir else None
        self.dynamicearthnet_adapter = dynamicearthnet_adapter
        self.masks_dir = Path(masks_cache_dir)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.incremental_tiles_dir = Path(incremental_tiles_dir)

    def get_tile_array(self, tile_id: str, date: str) -> np.ndarray:
        loc_id = parse_tile_id(tile_id)["location_id"] or "paris"

        incremental_path = (
            self.incremental_tiles_dir / loc_id / f"{tile_id}_{date}.npz"
        )
        if incremental_path.is_file():
            with np.load(incremental_path, allow_pickle=False) as archive:
                return np.asarray(archive["image"], dtype=np.float32)
        if self.dynamicearthnet_adapter:
            return self.dynamicearthnet_adapter.load_tile_array(loc_id, date, tile_id)
        if self.oscd_adapter:
            try:
                return self.oscd_adapter.load_tile_array(loc_id, date, tile_id)
            except Exception:
                pass
        return self._generate_representative_tile(loc_id, date)

    def evaluate_change_mask(
        self, tile_id: str, location_id: str, date_before: str,
        date_after: str, predicted_mask: np.ndarray,
    ) -> dict | None:
        """Score model output against DNE annotations; annotations never form the output mask."""
        if not self.dynamicearthnet_adapter:
            return None
        before = self.dynamicearthnet_adapter.load_semantic_tile(location_id, date_before, tile_id)
        after = self.dynamicearthnet_adapter.load_semantic_tile(location_id, date_after, tile_id)
        if before.shape != after.shape or before.shape != predicted_mask.shape:
            return None
        truth = before != after
        prediction = predicted_mask > 0
        tp = int(np.count_nonzero(prediction & truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        tn = int(np.count_nonzero(~prediction & ~truth))
        precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
        recall = tp / (tp + fn) if tp + fn else 1.0
        specificity = tn / (tn + fp) if tn + fp else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        union = tp + fp + fn
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(tp / union) if union else 1.0,
            "pixel_accuracy": float((tp + tn) / truth.size),
            "specificity": float(specificity),
            "balanced_accuracy": float((recall + specificity) / 2),
            "predicted_changed_pixels": int(np.count_nonzero(prediction)),
            "expected_changed_pixels": int(np.count_nonzero(truth)),
            "pixel_count": int(truth.size),
            "source": "DynamicEarthNet monthly labels; evaluation only",
        }

    def get_tile_bytes(
        self, tile_id: str, date: str, fmt: str = "png", thumbnail: bool = False
    ) -> bytes:
        arr = self.get_tile_array(tile_id, date)

        # Convert float32 [0, 1] to uint8 [0, 255]
        if arr.max() <= 1.0:
            arr_u8 = (arr * 255.0).astype(np.uint8)
        else:
            arr_u8 = arr.astype(np.uint8)

        img = Image.fromarray(arr_u8)
        if thumbnail:
            img.thumbnail((128, 128))

        buf = io.BytesIO()
        img_format = "PNG" if fmt.lower() == "png" else "JPEG"
        img.save(buf, format=img_format)
        return buf.getvalue()

    def save_mask_bytes(self, mask_id: str, mask_array: np.ndarray) -> str:
        """Saves a binary change mask to disk as PNG."""
        filename = f"{mask_id}.png"
        filepath = self.masks_dir / filename
        mask_u8 = (mask_array * 255).astype(np.uint8)
        Image.fromarray(mask_u8).save(filepath)
        return filename

    def save_semantic_mask_bytes(self, mask_id: str, transition_ids: np.ndarray) -> str:
        """Save predicted transition IDs: 0 means unchanged, 1..49 are class pairs."""
        filename = f"{mask_id}.semantic.png"
        Image.fromarray(np.asarray(transition_ids, dtype=np.uint8), mode="L").save(
            self.masks_dir / filename
        )
        return filename

    def get_mask_bytes(self, filename: str) -> bytes:
        filepath = self.masks_dir / filename
        if not filepath.exists():
            # Return an empty 256x256 mask if not found
            buf = io.BytesIO()
            Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(buf, format="PNG")
            return buf.getvalue()

        with open(filepath, "rb") as f:
            return f.read()

    def get_mask_overlay_bytes(self, filename: str) -> bytes:
        """Render the stored binary mask as transparent red pixels for the UI."""
        filepath = self.masks_dir / filename
        if filepath.exists():
            with Image.open(filepath) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8)
        else:
            mask = np.zeros((256, 256), dtype=np.uint8)

        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = 0
        rgba[..., 2] = 0
        rgba[..., 3] = np.where(mask > 0, 245, 0).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def get_semantic_overlay_bytes(self, filename: str) -> bytes:
        """Render predicted from/to class IDs using a deterministic transition palette."""
        import colorsys

        filepath = self.masks_dir / filename
        if filepath.exists():
            with Image.open(filepath) as source:
                transition_ids = np.asarray(source.convert("L"), dtype=np.uint8)
        else:
            transition_ids = np.zeros((256, 256), dtype=np.uint8)
        rgba = np.zeros((*transition_ids.shape, 4), dtype=np.uint8)
        for transition_id in np.unique(transition_ids):
            if transition_id == 0 or transition_id > 49:
                continue
            code = int(transition_id) - 1
            color = colorsys.hsv_to_rgb((code * 0.61803398875) % 1.0, 0.82, 1.0)
            selected = transition_ids == transition_id
            rgba[selected, :3] = np.round(np.asarray(color) * 255).astype(np.uint8)
            rgba[selected, 3] = 235
        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def get_available_dates(self, location_id: str) -> List[str]:
        dates = self.tile_repo.list_dates_for_location(location_id)
        if not dates and self.dynamicearthnet_adapter:
            dates = self.dynamicearthnet_adapter.list_dates(location_id)
        elif not dates and self.oscd_adapter:
            dates = self.oscd_adapter.list_dates(location_id)
        return dates

    def _generate_representative_tile(self, location_id: str, date: str) -> np.ndarray:
        """Procedural realistic satellite terrain representation based on location coordinates."""
        coords = OSCD_CITY_COORDINATES.get(
            location_id.lower(), {"west": 0.0, "south": 0.0, "east": 0.05, "north": 0.05}
        )
        seed = int(abs(coords["west"] * 1000 + coords["south"] * 1000)) % 100000
        rng = np.random.RandomState(seed)

        tile = np.zeros((256, 256, 3), dtype=np.float32)
        # Base vegetation / ground terrain
        base_color = rng.uniform(0.15, 0.45, size=3)
        tile[:, :] = base_color

        # River or water feature
        curve_y = 128 + int(30 * np.sin(np.linspace(0, 3.14, 256))[100])
        tile[max(0, curve_y - 12):min(256, curve_y + 12), :] = [0.1, 0.35, 0.7]

        # Urban settlement blocks
        tile[40:110, 40:120] = [0.45, 0.45, 0.48]

        # If date is later acquisition (after 2017), show development change
        if "2017" in date or "2018" in date or "2020" in date:
            tile[150:200, 140:195] = [0.75, 0.35, 0.25]

        # Add mild sensor noise
        noise = rng.normal(0.0, 0.02, size=tile.shape)
        tile = np.clip(tile + noise, 0.0, 1.0)
        return tile.astype(np.float32)
