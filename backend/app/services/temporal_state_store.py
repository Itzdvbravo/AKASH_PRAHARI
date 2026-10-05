"""HDF5 persistence for Mamba's per-tile temporal feature/state snapshots."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


class TemporalStateStore:
    """Store compact per-date Mamba contexts, isolated by model fingerprint."""

    _lock = threading.RLock()

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _group_path(tile_id: str, date: str) -> str:
        tile_key = hashlib.sha256(tile_id.encode("utf-8")).hexdigest()
        safe_date = date.replace("/", "_").replace("\\", "_")
        return f"tiles/{tile_key}/{safe_date}"

    def load(
        self,
        tile_id: str,
        date: str,
        model_fingerprint: str,
    ) -> Optional[dict[str, np.ndarray]]:
        if not self.path.is_file():
            return None
        group_path = self._group_path(tile_id, date)
        with self._lock, h5py.File(self.path, "r") as store:
            if group_path not in store:
                return None
            group = store[group_path]
            if group.attrs.get("tile_id") != tile_id or group.attrs.get("date") != date:
                return None
            if group.attrs.get("model_fingerprint") != model_fingerprint:
                return None
            required = ("state", "conv", "spatial", "temporal")
            if any(name not in group for name in required):
                return None
            return {name: np.asarray(group[name], dtype=np.float32) for name in required}

    def save(
        self,
        tile_id: str,
        date: str,
        model_fingerprint: str,
        context: dict[str, np.ndarray],
    ) -> None:
        group_path = self._group_path(tile_id, date)
        with self._lock, h5py.File(self.path, "a") as store:
            if group_path in store:
                del store[group_path]
            group = store.create_group(group_path)
            group.attrs["tile_id"] = tile_id
            group.attrs["date"] = date
            group.attrs["model_fingerprint"] = model_fingerprint
            for name in ("state", "conv", "spatial", "temporal"):
                if name not in context:
                    raise ValueError(f"Temporal context is missing required field '{name}'")
                array = np.asarray(context[name], dtype=np.float16)
                group.create_dataset(name, data=array, compression="gzip", shuffle=True)
            store.flush()
