"""Alignment check utilities for backend comparison service."""
import numpy as np


def check_bitemporal_alignment(tile_a: np.ndarray, tile_b: np.ndarray) -> bool:
    if tile_a.shape != tile_b.shape:
        return False
    return True
