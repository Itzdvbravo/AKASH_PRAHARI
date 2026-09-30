"""Coordinate Reference System (CRS) transformations and bounding box utilities."""
from typing import Dict, Tuple

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


def normalize_crs(crs_string: str) -> str:
    """Standardizes CRS string representation (e.g. EPSG:4326)."""
    crs_clean = crs_string.strip().upper()
    if crs_clean.startswith("4326") or crs_clean == "WGS84":
        return "EPSG:4326"
    if not crs_clean.startswith("EPSG:"):
        return f"EPSG:{crs_clean}"
    return crs_clean


def transform_bounds(
    bounds: Dict[str, float],
    src_crs: str,
    dst_crs: str = "EPSG:4326"
) -> Dict[str, float]:
    """Transform bounding box from src_crs to dst_crs."""
    src_norm = normalize_crs(src_crs)
    dst_norm = normalize_crs(dst_crs)

    if src_norm == dst_norm or not HAS_PYPROJ:
        return bounds

    transformer = Transformer.from_crs(src_norm, dst_norm, always_xy=True)
    min_x, min_y = transformer.transform(bounds["west"], bounds["south"])
    max_x, max_y = transformer.transform(bounds["east"], bounds["north"])

    return {
        "west": float(min_x),
        "south": float(min_y),
        "east": float(max_x),
        "north": float(max_y)
    }
