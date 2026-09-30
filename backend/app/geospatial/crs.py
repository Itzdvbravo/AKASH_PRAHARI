"""Geospatial CRS conversion and bounding box utilities for backend."""
from typing import Dict


def normalize_crs(crs_string: str) -> str:
    crs_clean = crs_string.strip().upper()
    if crs_clean.startswith("4326") or crs_clean == "WGS84":
        return "EPSG:4326"
    if not crs_clean.startswith("EPSG:"):
        return f"EPSG:{crs_clean}"
    return crs_clean


def bbox_to_geojson_polygon(bbox: Dict[str, float]) -> Dict:
    """Converts {west, south, east, north} to GeoJSON Polygon dict."""
    w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    return {
        "type": "Polygon",
        "coordinates": [[
            [w, s],
            [e, s],
            [e, n],
            [w, n],
            [w, s]
        ]]
    }
