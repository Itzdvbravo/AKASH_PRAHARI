"""Integration API tests for TerraEyes endpoints."""
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "embedding_model" in data
    assert "change_detector" in data


def test_search_endpoint(client):
    payload = {
        "query_type": "text",
        "query_text": "urban construction near water",
        "top_k": 5
    }
    res = client.post("/api/v1/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "query_id" in data
    assert "results" in data
    assert len(data["results"]) <= 5
    if len(data["results"]) > 0:
        first = data["results"][0]
        assert "tile_ref" in first
        assert "confidence" in first
        assert "geo_bbox" in first


def test_image_and_dates_endpoints(client):
    dates_res = client.get("/api/v1/images/paris/dates")
    assert dates_res.status_code == 200
    dates_data = dates_res.json()
    assert "dates" in dates_data
    assert len(dates_data["dates"]) >= 2

    # Fetch image
    img_res = client.get("/api/v1/images/paris_0001_0001_sentinel-2/2016-03-15")
    assert img_res.status_code == 200
    assert img_res.headers["content-type"] == "image/png"
    assert len(img_res.content) > 100


def test_comparison_and_change_detection_flow(client):
    comp_payload = {
        "location_id": "paris",
        "tile_id": "paris_0001_0001_sentinel-2",
        "date_before": "2016-03-15",
        "date_after": "2018-06-20",
        "sensor": "sentinel-2"
    }
    # Test both /comparison and /compare endpoints
    comp_res = client.post("/api/v1/compare", json=comp_payload)
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert "comparison_id" in comp_data
    comp_id = comp_data["comparison_id"]

    # Trigger change detection with comparison_id
    cd_payload = {"comparison_id": comp_id}
    cd_res = client.post("/api/v1/change-detection", json=cd_payload)
    assert cd_res.status_code == 200
    cd_data = cd_res.json()
    assert cd_data["status"] == "completed"
    assert "mask_url" in cd_data
    assert "summary" in cd_data
    assert "bounding_boxes" in cd_data

    # Fetch mask image
    mask_url = cd_data["mask_url"]
    mask_res = client.get(mask_url)
    assert mask_res.status_code == 200
    assert mask_res.headers["content-type"] == "image/png"


def test_search_empty_query_and_all_locations(client):
    """Verifies that frontend's initial load pattern handleSearch('', 'any', 'all') succeeds."""
    payload = {
        "query_type": "text",
        "query_text": "",
        "filters": {
            "sensor": "any",
            "location": "all"
        }
    }
    res = client.post("/api/v1/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) > 0


def test_change_detection_auto_dates(client):
    """Verifies change detection works when dates are omitted and resolved from catalog."""
    payload = {
        "location_id": "paris",
        "tile_id": "paris_0001_0001_sentinel-2"
    }
    res = client.post("/api/v1/change-detection", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["summary"]["location"] == "Paris"
