"""Unit tests for OSCDLoader, OSCDMetadataExtractor, and OSCDAdapter."""
from pathlib import Path
from adapters.oscd.oscd_metadata import OSCDMetadataExtractor, OSCD_CITY_COORDINATES
from adapters.oscd.oscd_adapter import OSCDAdapter
from geospatial.tile_id import build_tile_id, parse_tile_id


def test_metadata_extractor_coordinates():
    extractor = OSCDMetadataExtractor()
    paris_bbox = extractor.get_geo_bbox("paris")
    assert paris_bbox["north"] > paris_bbox["south"]
    assert paris_bbox["east"] > paris_bbox["west"]
    assert "paris" in OSCD_CITY_COORDINATES


def test_tile_id_parsing():
    tid = build_tile_id("lasvegas", 2, 5, "sentinel-2")
    assert tid == "lasvegas_0002_0005_sentinel-2"

    parsed = parse_tile_id(tid)
    assert parsed["location_id"] == "lasvegas"
    assert parsed["row"] == "0002"
    assert parsed["col"] == "0005"
    assert parsed["sensor"] == "sentinel-2"


def test_oscd_adapter_city_list(tmp_path):
    adapter = OSCDAdapter(str(tmp_path))
    cities = adapter.list_locations()
    assert len(cities) >= 20
    assert "paris" in cities
    assert "berlin" in cities

    dates = adapter.list_dates("paris")
    assert len(dates) >= 2
