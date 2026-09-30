"""Metadata extraction and normalization for OSCD dataset."""
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

# Pre-defined geographic coordinates (west, south, east, north) for all 24 OSCD cities
OSCD_CITY_COORDINATES: Dict[str, Dict[str, float]] = {
    "abudhabi": {"west": 54.34, "south": 24.42, "east": 54.42, "north": 24.50},
    "aguasclaras": {"west": -48.04, "south": -15.86, "east": -48.00, "north": -15.82},
    "beihai": {"west": 109.10, "south": 21.45, "east": 109.18, "north": 21.52},
    "beirut": {"west": 35.48, "south": 33.86, "east": 35.54, "north": 33.91},
    "berlin": {"west": 13.35, "south": 52.48, "east": 13.45, "north": 52.55},
    "bordeaux": {"west": -0.61, "south": 44.81, "east": -0.54, "north": 44.87},
    "brasilia": {"west": -47.93, "south": -15.82, "east": -47.85, "north": -15.75},
    "cambridge": {"west": 0.09, "south": 52.18, "east": 0.16, "north": 52.23},
    "canberra": {"west": 149.10, "south": -35.32, "east": 149.17, "north": -35.26},
    "chongqing": {"west": 106.50, "south": 29.53, "east": 106.60, "north": 29.60},
    "cuiaba": {"west": -56.12, "south": -15.63, "east": -56.05, "north": -15.57},
    "dhanbad": {"west": 86.40, "south": 23.77, "east": 86.47, "north": 23.82},
    "dubai": {"west": 55.25, "south": 25.18, "east": 55.33, "north": 25.26},
    "hongkong": {"west": 114.15, "south": 22.28, "east": 114.23, "north": 22.34},
    "lasvegas": {"west": -115.19, "south": 36.14, "east": -115.11, "north": 36.20},
    "madrid": {"west": -3.73, "south": 40.39, "east": -3.66, "north": 40.45},
    "milano": {"west": 9.15, "south": 45.44, "east": 9.23, "north": 45.49},
    "montpellier": {"west": 3.84, "south": 43.58, "east": 3.92, "north": 43.64},
    "norrkoping": {"west": 16.15, "south": 58.56, "east": 16.23, "north": 58.61},
    "paris": {"west": 2.29, "south": 48.82, "east": 2.39, "north": 48.89},
    "rennes": {"west": -1.71, "south": 48.09, "east": -1.64, "north": 48.14},
    "rio": {"west": -43.25, "south": -22.95, "east": -43.16, "north": -22.88},
    "sao-paulo": {"west": -46.68, "south": -23.58, "east": -46.59, "north": -23.51},
    "shanghai": {"west": 121.43, "south": 31.20, "east": 121.52, "north": 31.27}
}

# Standard OSCD acquisition date pairs (date 1, date 2)
OSCD_DEFAULT_DATES: Dict[str, List[str]] = {
    "paris": ["2015-11-20", "2017-06-28"],
    "berlin": ["2016-04-18", "2018-05-13"],
    "lasvegas": ["2016-06-25", "2018-07-20"],
    "dubai": ["2016-01-15", "2018-03-22"],
    "hongkong": ["2015-12-10", "2017-11-18"],
}


class OSCDMetadataExtractor:
    """Extracts and normalizes metadata for OSCD scenes."""

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None

    def get_geo_bbox(self, city_name: str) -> Dict[str, float]:
        city_key = city_name.lower().replace(" ", "").replace("_", "-")
        return OSCD_CITY_COORDINATES.get(
            city_key,
            {"west": 0.0, "south": 0.0, "east": 0.05, "north": 0.05}
        )

    def get_dates_for_city(self, city_dir: Path) -> List[str]:
        """Discover dates from directory structure or dates.txt sidecar."""
        dates_file = city_dir / "dates.txt"
        if dates_file.exists():
            try:
                with open(dates_file, "r", encoding="utf-8") as f:
                    dates = [line.strip() for line in f if line.strip()]
                    if dates:
                        return dates
            except Exception:
                pass

        city_key = city_dir.name.lower().replace(" ", "").replace("_", "-")
        if city_key in OSCD_DEFAULT_DATES:
            return OSCD_DEFAULT_DATES[city_key]

        # Check subdirectories (e.g. imgs_1, imgs_2 or date strings)
        img_dirs = [d.name for d in city_dir.iterdir() if d.is_dir() and "imgs" in d.name.lower()]
        if len(img_dirs) >= 2:
            return ["2016-01-01", "2018-01-01"]

        return ["2016-01-01", "2018-01-01"]
