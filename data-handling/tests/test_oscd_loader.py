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


def test_metadata_extractor_parses_oscd_date_sidecar(tmp_path):
    city_dir = tmp_path / "paris"
    city_dir.mkdir()
    (city_dir / "dates.txt").write_text(
        "date_1: 20161130\ndate_2: 20171107\n", encoding="utf-8"
    )

    extractor = OSCDMetadataExtractor(tmp_path)
    assert extractor.get_dates_for_city(city_dir) == ["2016-11-30", "2017-11-07"]


def test_metadata_extractor_prefers_city_geojson_bounds(tmp_path):
    city_dir = tmp_path / "paris"
    city_dir.mkdir()
    (city_dir / "paris.geojson").write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"geometry":{"type":"Polygon","coordinates":[[[2.1,48.1],'
        '[2.4,48.1],[2.4,48.5],[2.1,48.5],[2.1,48.1]]]}}]}',
        encoding="utf-8",
    )

    bbox = OSCDMetadataExtractor(tmp_path).get_geo_bbox("paris")
    assert bbox == {"west": 2.1, "south": 48.1, "east": 2.4, "north": 48.5}


def test_tile_id_parsing():
    tid = build_tile_id("lasvegas", 2, 5, "sentinel-2")
    assert tid == "lasvegas_0002_0005_sentinel-2"

    parsed = parse_tile_id(tid)
    assert parsed["location_id"] == "lasvegas"
    assert parsed["row"] == "0002"
    assert parsed["col"] == "0005"
    assert parsed["sensor"] == "sentinel-2"

    underscored_city = parse_tile_id("saclay_e_0003_0005_sentinel-2")
    assert underscored_city == {
        "location_id": "saclay_e",
        "row": "0003",
        "col": "0005",
        "sensor": "sentinel-2",
    }


def test_oscd_adapter_city_list(tmp_path):
    adapter = OSCDAdapter(str(tmp_path))
    cities = adapter.list_locations()
    assert len(cities) >= 20
    assert "paris" in cities
    assert "berlin" in cities

    dates = adapter.list_dates("paris")
    assert len(dates) >= 2
