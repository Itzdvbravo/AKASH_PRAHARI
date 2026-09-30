"""Tiling and reconstruction functions with persistent tile IDs."""
from typing import Dict, List, Tuple
import numpy as np


def generate_tile_id(location_id: str, row: int, col: int, sensor: str = "sentinel-2") -> str:
    """Generate canonical persistent geographic tile identifier."""
    return f"{location_id.lower()}_{row:04d}_{col:04d}_{sensor.lower()}"


def tile_scene(
    scene_array: np.ndarray,
    tile_size: int = 256,
    stride: int = 256,
    location_id: str = "loc",
    sensor: str = "sentinel-2"
) -> List[Tuple[str, np.ndarray, Tuple[int, int, int, int]]]:
    """
    Slices (H, W, C) scene into tiles of shape (tile_size, tile_size, C).
    Returns list of tuples: (tile_id, tile_array, (row_start, row_end, col_start, col_end)).
    """
    h, w = scene_array.shape[:2]
    c = scene_array.shape[2] if scene_array.ndim == 3 else 1
    tiles = []

    row_idx = 0
    for r in range(0, h, stride):
        col_idx = 0
        r_end = min(r + tile_size, h)
        if r_end - r < tile_size // 2:
            continue

        for col in range(0, w, stride):
            c_end = min(col + tile_size, w)
            if c_end - col < tile_size // 2:
                continue

            tile_id = generate_tile_id(location_id, row_idx, col_idx, sensor)

            # Pad if boundary tile
            if scene_array.ndim == 3:
                tile = np.zeros((tile_size, tile_size, c), dtype=scene_array.dtype)
                tile[0:(r_end - r), 0:(c_end - col), :] = scene_array[r:r_end, col:c_end, :]
            else:
                tile = np.zeros((tile_size, tile_size), dtype=scene_array.dtype)
                tile[0:(r_end - r), 0:(c_end - col)] = scene_array[r:r_end, col:c_end]

            tiles.append((tile_id, tile, (r, r_end, col, c_end)))
            col_idx += 1
        row_idx += 1

    return tiles


def assemble_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    output_shape: Tuple[int, int]
) -> np.ndarray:
    """Reassembles tiles back to original scene dimensions."""
    if not tiles:
        return np.zeros(output_shape, dtype=np.float32)

    sample = tiles[0][0]
    h, w = output_shape[:2]
    if sample.ndim == 3:
        c = sample.shape[2]
        canvas = np.zeros((h, w, c), dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)
    else:
        canvas = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)

    for tile_arr, (r_start, r_end, c_start, c_end) in tiles:
        tile_h = r_end - r_start
        tile_w = c_end - c_start
        if sample.ndim == 3:
            canvas[r_start:r_end, c_start:c_end, :] += tile_arr[0:tile_h, 0:tile_w, :]
            weights[r_start:r_end, c_start:c_end, :] += 1.0
        else:
            canvas[r_start:r_end, c_start:c_end] += tile_arr[0:tile_h, 0:tile_w]
            weights[r_start:r_end, c_start:c_end] += 1.0

    weights = np.where(weights == 0, 1.0, weights)
    return canvas / weights
