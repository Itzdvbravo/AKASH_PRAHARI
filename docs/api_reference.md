# TerraEyes API Specification (v1)

Base URL: `/api/v1`

## Endpoints Summary

| Method | Endpoint | Description | Phase |
|--------|----------|-------------|-------|
| GET | `/health` | Service health status and system configuration | Prototype |
| POST | `/search` | Submit text or image query for semantic retrieval | Prototype |
| GET | `/images/{tile_id}/{date}` | Retrieve image tile bytes for location and date | Prototype |
| GET | `/images/{location_id}/dates` | List available temporal dates for location | Prototype |
| POST | `/comparison` | Initiate bi-temporal image pair comparison | Prototype |
| POST | `/change-detection` | Run change detection analysis on image pair | Prototype |
| GET | `/similar-sites/{tile_id}` | Find cross-location embedding neighbors for discovery | Prototype |
| GET | `/review-queue` | List persisted change candidates with analyst decisions | Prototype |
| GET | `/review-queue/{job_id}` | Read a candidate and its review history | Prototype |
| POST | `/review-queue/{job_id}/reviews` | Append a confirm/reject decision to the audit trail | Prototype |
| GET | `/review-queue/export.geojson` | Export candidates, review history, and source provenance | Prototype |
| GET | `/masks/{mask_filename}` | Fetch binary change mask overlay PNG | Prototype |

---

## 1. GET `/health`
Returns runtime system status and active model configuration.

### Response 200 OK
```json
{
  "status": "ok",
  "version": "0.1.0",
  "embedding_model": "mock",
  "change_detector": "pixel_diff",
  "faiss_index_size": 5,
  "db_tiles": 5
}
```

## Analyst review and discovery

Every successful change-detection run is persisted to the review queue with its
tile/date/sensor, CRS, resolution, bounds, detector, mask URL, and detection
payload. `GET /review-queue?limit=100` lists candidates; `GET
/review-queue/{job_id}` returns a candidate and its review history. Append
feedback with `POST /review-queue/{job_id}/reviews`:

```json
{"decision": "confirmed", "comment": "Expansion visible in source pair", "reviewer": "analyst-1"}
```

The decision may be `confirmed` or `rejected`; reviews are append-only. They
are retained for audit but are not currently used to train or rerank results.
`GET /review-queue/export.geojson` exports WGS84 bounding-box features with
detection, processing provenance, and all review decisions.

`GET /similar-sites/{tile_id}?top_k=10` embeds the selected tile and returns
nearest indexed tiles from distinct locations. This is embedding-neighbor
discovery; it does not claim a separately trained site-clustering model.

---

## 2. POST `/search`
Performs semantic retrieval matching text or image query against indexed satellite imagery.

### Request Body
```json
{
  "query_type": "text",
  "query_text": "urban expansion near river",
  "query_image_b64": null,
  "filters": {
    "location": "paris",
    "date_from": "2018-01-01",
    "date_to": "2021-12-31",
    "sensor": "sentinel-2",
    "area_of_interest": {"west": 2.0, "south": 48.0, "east": 3.0, "north": 49.0}
  },
  "top_k": 10
}
```

`date_from`, `date_to`, and `area_of_interest` are applied to the catalogued
observation represented by each embedding. The AOI is a WGS84 bounding box and
retains tiles whose bounds intersect it.

### Response 200 OK
```json
{
  "query_id": "8e3c4a21-9951-4b10-8b1e-0123456789ab",
  "results": [
    {
      "rank": 1,
      "tile_ref": {
        "tile_id": "paris_0001_0001_sentinel-2",
        "location_id": "paris",
        "date": "2020-06-15",
        "sensor": "sentinel-2"
      },
      "confidence": {
        "score": 0.87,
        "method": "cosine_similarity",
        "calibrated": false
      },
      "thumbnail_url": "/api/v1/images/paris_0001_0001_sentinel-2/2020-06-15",
      "available_dates": ["2018-03-10", "2020-06-15"],
      "geo_bbox": {
        "west": 2.29,
        "south": 48.85,
        "east": 2.31,
        "north": 48.87
      }
    }
  ],
  "total": 1,
  "query_ms": 42
}
```

---

## 3. POST `/comparison`
Registers a bi-temporal comparison pairing between two acquisition dates for a given location tile.

### Request Body
```json
{
  "location_id": "paris",
  "tile_id": "paris_0001_0001_sentinel-2",
  "date_before": "2018-03-10",
  "date_after": "2020-06-15",
  "sensor": "sentinel-2"
}
```

### Response 200 OK
```json
{
  "comparison_id": "c92f1b40-1234-5678-9abc-def012345678",
  "before": {
    "tile_ref": {
      "tile_id": "paris_0001_0001_sentinel-2",
      "location_id": "paris",
      "date": "2018-03-10",
      "sensor": "sentinel-2"
    },
    "image_url": "/api/v1/images/paris_0001_0001_sentinel-2/2018-03-10",
    "cloud_cover_pct": 0.0
  },
  "after": {
    "tile_ref": {
      "tile_id": "paris_0001_0001_sentinel-2",
      "location_id": "paris",
      "date": "2020-06-15",
      "sensor": "sentinel-2"
    },
    "image_url": "/api/v1/images/paris_0001_0001_sentinel-2/2020-06-15",
    "cloud_cover_pct": 0.0
  },
  "coregistered": true
}
```

---

## 4. POST `/change-detection`
Executes change detection algorithm on registered bi-temporal image pair and generates structured analyst summary.

### Request Body
```json
{
  "comparison_id": "c92f1b40-1234-5678-9abc-def012345678",
  "location_id": "paris",
  "tile_id": "paris_0001_0001_sentinel-2",
  "date_before": "2018-03-10",
  "date_after": "2020-06-15",
  "temporal_dates": ["2018-03-10", "2019-02-20", "2020-06-15"]
}
```

`temporal_dates` is optional and applies to the `mamba_cd` detector. It must be
unique, sorted ascending, and include both comparison endpoints. If omitted,
Mamba processes all dates recorded for the tile between `date_before` and
`date_after`, then persists its per-date state in HDF5. Pairwise baselines ignore
this field. OSCD currently supplies only two dates per scene.

### Response 200 OK
```json
{
  "job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d4e5",
  "status": "completed",
  "mask_url": "/api/v1/masks/f47ac10b-58cc-4372-a567-0e02b2c3d4e5.png",
  "bounding_boxes": [
    {
      "x": 40,
      "y": 60,
      "width": 50,
      "height": 50,
      "label": "change",
      "confidence": {
        "score": 0.82,
        "method": "pixel_fraction",
        "calibrated": false
      }
    }
  ],
  "summary": {
    "location": "Paris, France",
    "date_before": "2018-03-10",
    "date_after": "2020-06-15",
    "change_type": "binary_surface_change",
    "earliest_detectable_change": null,
    "confidence": {
      "score": 0.82,
      "method": "pixel_fraction",
      "calibrated": false
    },
    "changed_pixel_fraction": 0.098,
    "source_provenance": "Synthetic Fixture v0.1",
    "detector": "pixel_diff"
  },
  "processing_ms": 45
}
```
