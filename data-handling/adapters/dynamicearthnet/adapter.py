"""Lazy access to the compact DynamicEarthNet-video TACO archive."""
from __future__ import annotations

from collections import OrderedDict
from datetime import date, timedelta
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import h5py
import numpy as np

LULC_CLASSES = {
    0: "impervious surface",
    1: "agriculture",
    2: "forest and other vegetation",
    3: "wetlands",
    4: "bare soil",
    5: "water",
    6: "snow and ice",
}
IMAGE_SIZE = 1024
IMAGE_DN_SCALE = 10_000.0


def _asset_bytes(archive_path: Path, asset: dict[str, Any]) -> bytes:
    offset, size = int(asset["internal:offset"]), int(asset["internal:size"])
    with archive_path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(size)
    if len(payload) != size:
        raise IOError(f"Truncated DynamicEarthNet asset at offset {offset}")
    return payload


class DynamicEarthNetAdapter:
    """Decode monthly RGB and categorical label tiles from DynamicEarthNet-video."""

    def __init__(
        self,
        archive_path: str,
        cache_items: int = 6,
        ffmpeg_path: str | None = None,
        decoded_cache_dir: str | None = None,
    ):
        archive = Path(archive_path).expanduser()
        if not archive.is_file():
            raise FileNotFoundError(f"DynamicEarthNet archive not found: {archive}")
        try:
            import tacoreader
        except ImportError as exc:
            raise RuntimeError("DynamicEarthNet access requires tacoreader; install setup/requirements-dev.txt") from exc
        self.archive_path = archive.resolve()
        self.decoded_cache_dir = Path(decoded_cache_dir) if decoded_cache_dir else archive.parent / "decoded_frames"
        self.decoded_cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = tacoreader.load(str(self.archive_path))
        rows = self.dataset.data.to_arrow().to_pylist()
        self.areas = {str(row["id"]): row for row in rows}
        self.ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise RuntimeError("FFmpeg is required to decode DynamicEarthNet video assets")
        self.cache_items = max(2, cache_items)
        self._cache: OrderedDict[tuple[str, str, str], np.ndarray] = OrderedDict()
        self._asset_cache: dict[str, dict[str, Any]] = {}
        self._date_cache: dict[str, list[str]] = {}

    def list_locations(self) -> list[str]:
        return [f"den{area_id}" for area_id in sorted(self.areas)]

    def _area_id(self, location_id: str) -> str:
        area_id = location_id.lower().removeprefix("den")
        if area_id not in self.areas:
            raise KeyError(f"Unknown DynamicEarthNet area: {location_id}")
        return area_id

    def _assets(self, area_id: str) -> dict[str, Any]:
        if area_id not in self._asset_cache:
            assets = self.dataset.data.read(area_id).to_arrow().to_pylist()
            self._asset_cache[area_id] = {asset["id"]: asset for asset in assets}
        return self._asset_cache[area_id]

    def list_dates(self, location_id: str) -> list[str]:
        area_id = self._area_id(location_id)
        if area_id not in self._date_cache:
            asset = self._assets(area_id)["metadata"]
            with h5py.File(io.BytesIO(_asset_bytes(self.archive_path, asset)), "r") as metadata:
                offsets = np.asarray(metadata["time_month"], dtype=np.int64)
                units = metadata["time_month"].attrs.get("units", "days since 2018-01-01")
            if isinstance(units, bytes):
                units = units.decode("utf-8")
            origin = date.fromisoformat(str(units).split("since", 1)[-1].strip()[:10])
            self._date_cache[area_id] = [(origin + timedelta(days=int(offset))).isoformat() for offset in offsets]
        return self._date_cache[area_id]

    def get_area_georeferencing(self, location_id: str) -> tuple[str, list[float]]:
        row = self.areas[self._area_id(location_id)]
        return str(row["stac:crs"]), [float(value) for value in row["stac:geotransform"]]

    def _decode_month(self, area_id: str, acquisition_date: str, kind: str) -> np.ndarray:
        key = (area_id, acquisition_date, kind)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        cache_kind = f"{kind}-v2" if kind == "image" else kind
        cache_path = self.decoded_cache_dir / f"{area_id}_{acquisition_date}_{cache_kind}.npy"
        if cache_path.is_file():
            decoded = np.load(cache_path, allow_pickle=False)
            self._remember(key, decoded)
            return decoded
        dates = self.list_dates(f"den{area_id}")
        try:
            frame = dates.index(acquisition_date)
        except ValueError as exc:
            raise ValueError(f"No DynamicEarthNet monthly label for {acquisition_date}") from exc
        assets = self._assets(area_id)
        if kind == "image":
            offset = (date.fromisoformat(acquisition_date) - date(2018, 1, 1)).days
            asset = assets["bands_1"]
            pixel_format, dtype, channels, frame_indices = "rgb48le", np.dtype("<u2"), 3, [offset]
        else:
            asset = assets["labels"]
            pixel_format, dtype, channels, frame_indices = "bgr0", np.dtype("u1"), 4, [frame]
        command = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", "pipe:0",
                   "-vf", f"select=eq(n\\,{frame_indices[0]})", "-vsync", "0", "-f", "rawvideo",
                   "-pix_fmt", pixel_format, "pipe:1"]
        completed = subprocess.run(command, input=_asset_bytes(self.archive_path, asset), capture_output=True,
                                   check=True, timeout=900)
        expected = IMAGE_SIZE * IMAGE_SIZE * channels * dtype.itemsize
        if len(completed.stdout) != expected:
            raise ValueError(f"Expected one decoded {kind} frame for {area_id}/{acquisition_date}")
        raw = np.frombuffer(completed.stdout, dtype=dtype).reshape(IMAGE_SIZE, IMAGE_SIZE, channels)
        if kind == "image":
            # A per-frame percentile scale changes radiometry from month to
            # month and creates artificial temporal differences. Use the same
            # PlanetFusion DN scale for every acquisition instead.
            decoded = np.clip(raw.astype(np.float32) / IMAGE_DN_SCALE, 0.0, 1.0)
        else:
            rgb = raw[..., :3]
            if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
                raise ValueError("DynamicEarthNet label frame is not an indexed grayscale class map")
            decoded = rgb[..., 0].astype(np.uint8)
            if int(decoded.max(initial=0)) > max(LULC_CLASSES):
                raise ValueError("DynamicEarthNet label contains an unknown class ID")
        self._remember(key, decoded)
        self._persist(cache_path, decoded)
        return decoded

    def _remember(self, key: tuple[str, str, str], decoded: np.ndarray) -> None:
        self._cache[key] = decoded
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_items:
            self._cache.popitem(last=False)

    @staticmethod
    def _persist(path: Path, decoded: np.ndarray) -> None:
        """Keep decoded frames across requests and API restarts."""
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy.tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            np.save(stream, decoded, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)

    @staticmethod
    def _tile_bounds(tile_id: str) -> tuple[str, int, int]:
        parts = tile_id.split("_")
        if len(parts) < 4 or not parts[-3].isdigit() or not parts[-2].isdigit():
            raise ValueError(f"Invalid DynamicEarthNet tile ID: {tile_id}")
        return "_".join(parts[:-3]), int(parts[-3]), int(parts[-2])

    def load_tile_array(self, location_id: str, acquisition_date: str, tile_id: str) -> np.ndarray:
        tile_location, row, col = self._tile_bounds(tile_id)
        if tile_location.lower() != location_id.lower():
            raise ValueError("Tile ID location does not match requested DynamicEarthNet area")
        image = self._decode_month(self._area_id(location_id), acquisition_date, "image")
        r, c = row * 256, col * 256
        if r >= IMAGE_SIZE or c >= IMAGE_SIZE:
            raise ValueError(f"DynamicEarthNet tile index is outside the source area: {tile_id}")
        return image[r:r + 256, c:c + 256].copy()

    def load_semantic_tile(self, location_id: str, acquisition_date: str, tile_id: str) -> np.ndarray:
        tile_location, row, col = self._tile_bounds(tile_id)
        if tile_location.lower() != location_id.lower():
            raise ValueError("Tile ID location does not match requested DynamicEarthNet area")
        labels = self._decode_month(self._area_id(location_id), acquisition_date, "labels")
        r, c = row * 256, col * 256
        if r >= IMAGE_SIZE or c >= IMAGE_SIZE:
            raise ValueError(f"DynamicEarthNet tile index is outside the source area: {tile_id}")
        return labels[r:r + 256, c:c + 256].copy()

    def close(self) -> None:
        self.dataset.close()
