# TerraEyes — Extensive Implementation Plan

---

## 1. Understanding of the Reference Architecture and Application Requirements

### 1.1 System Summary

TerraEyes is a **fully offline**, analyst-oriented geospatial intelligence platform that combines:

- **Semantic retrieval** of satellite imagery via natural-language or image queries.
- **Multi-temporal change detection** using Mamba-based spatio-temporal models.
- **False-alarm suppression** to distinguish real land-cover changes from seasonal/sensor artefacts.
- **Discovery and clustering** of visually similar geographic locations.
- **Structured analyst output** including ranked results, before/after imagery, bounding boxes, confidence scores, timestamps, and source provenance.

### 1.2 Four-Pipeline Architecture (from reference images)

#### Pipeline 1 — Image Pre-Processing Pipeline
```
Input
  └─► Sensor Identification
        ├─► Landsat   → Optical calibration + cloud masking
        ├─► Sentinel-1 → Radar calibration + speckle + terrain processing
        └─► Sentinel-2 → Optical calibration + cloud masking
  └─► Co-registration + Quality Metadata  (← Last processed satellite-specific image fed back)
  └─► Tiling & Alignment  (Persistent geographic IDs w.r.t. location and sensor)
  └─► Output (pre-processed tiles with stable tile IDs)
```

#### Pipeline 2 — Image Processing Pipeline
```
Pre-processed tiles
  └─► Image Preprocessing Pipeline
        ├─► Global Embeddings ──► Merging Tiles ──► Vector Database
        └─► Dense Embeddings
              ├─► Spatio-Feature Extraction (Mamba) ──► Spatio Database (last tile)
              └─► Temporal Feature Extraction (Mamba) ◄──► Temporal Database
                    └─► Change Detector (Mamba)
                          └─► [IF DETECTED] Semantic Change Detection (Mamba)
                                └─► Merging Tiles ──► Spatio-Temporal Database
```
*State loops: Temporal DB stores previous and previous-projected Mamba states, updated after every Change Detector run regardless of outcome.*

#### Pipeline 3 — Querying Pipeline
```
Text Query
  └─► Text Preprocessing
        ├─► Semantic Info Extraction ──► Global Embeddings (text) ──► Spatio-Temporal DB ──► Ranking + Confidence ──► Results
        ├─► Location Info Extraction ──────────────────────────────► Spatio-Temporal DB
        └─► Temporal Info Extraction ──────────────────────────────► Spatio-Temporal DB

Image Query
  └─► Tiling & Sensor Identification ──► Global Embeddings (image) ──► Vector DB ──► Ranking + Confidence ──► Results
```

#### Pipeline 4 — Analyst Output (from reference image 4)
Core requirements confirmed from the reference image:
1. **Semantic Retrieval** — text or image query with location/date/sensor filters.
2. **Change Detection** — construction, demolition, road development, water-level changes.
3. **False-Alarm Suppression** — seasonal changes, clouds, shadows, haze, sensor differences, misalignment.
4. **Discovery and Clustering** — group similar locations, find comparable areas.
5. **Analyst Workflow** — ranked results with before/after imagery, confidence scores, geolocation, timestamps, source provenance; analyst feedback and auditing.
6. **Scalability and Incremental Ingestion** — add new scenes without full index rebuild; GeoTIFF/COG support.

**Hard Constraints:**
- Must operate **entirely offline** after setup (no cloud services, no external APIs at runtime).
- Public pretrained models allowed — licences and origins must be documented, weights stored locally.
- Must demonstrate retrieval and change detection on **public imagery and unseen evaluation cases**.

---

## 2. Explicit Assumptions, Ambiguities, and Questions Requiring Clarification

### 2.1 Confirmed Assumptions
- OSCD is the initial dataset. Its directory structure, bands, and metadata must be verified once downloaded.
- The Mamba architecture referenced in the image diagrams is the **Selective State Space Model (SSM)** (Gu & Dao, 2023). PyPI package: `mamba-ssm`.
- "Vector Database" = embedding index for image-level global embeddings (candidate: FAISS or ChromaDB, offline).
- "Spatio Database" = per-tile spatial feature store (last processed image tile per location).
- "Temporal Database" = recurrent Mamba hidden state snapshots (previous and previous-projected states per location).
- "Spatio-Temporal Database" = merged semantic change detection outputs, queryable by text.
- The "Merging Tiles" step reassembles per-tile results back to scene-level representations.
- Persistent geographic tile IDs must be consistent across sensors and dates.

### 2.2 Open Questions / Ambiguities (require clarification before locking certain modules)

| # | Question | Impact |
|---|----------|--------|
| Q1 | What is the expected hardware baseline for inference? (CPU only? GPU with VRAM limit?) | Determines which Mamba variant and embedding model sizes are feasible. |
| Q2 | Is the "Temporal Database" storing full hidden states or compressed projections? | Storage design for large time series. |
| Q3 | Should the Spatio-Temporal Database support geographic bounding-box queries (spatial index)? | Determines whether to use PostGIS, SQLite-Spatialite, or a flat GeoJSON store. |
| Q4 | Is analyst feedback (thumbs up/down on results) required in the prototype or Phase B? | Frontend/backend complexity. |
| Q5 | Are bounding boxes derived from connected-component analysis of the change mask, or from a separate object-detection head? | Model complexity. |
| Q6 | Should the application support concurrent multi-user sessions, or is single-user local operation sufficient? | API server and state management. |
| Q7 | What tile size is expected (e.g., 256×256, 512×512)? | Memory and throughput. |
| Q8 | Does "offline after setup" include downloading model weights at install time, or must weights already be on disk before any network access? | Setup documentation. |
| Q9 | Are Sentinel-1, Sentinel-2, and Landsat required for Phase A or only Phase B? | Scope boundary. |
| Q10 | Is the "confidence score" a calibrated probability, a cosine similarity, or a heuristic blend? | Output schema. |

### 2.3 Items That Must Be Verified When OSCD Becomes Available
- Band composition (OSCD uses Sentinel-2 bands B01–B12 plus B8A).
- Directory structure (city/date/band TIFF organisation).
- Available ground-truth change masks (binary rasters).
- Coordinate reference system and resolution (10 m / 20 m / 60 m bands).
- Whether co-registration has already been applied.
- Available metadata files (JSON, XML, or sidecar).
- City list and geographic diversity.
- Licence terms for redistribution and derivative works.

---

## 3. Proposed Technology Stack and Justification

### 3.1 Backend
| Component | Choice | Justification |
|-----------|--------|---------------|
| Language | Python 3.11 | Ecosystem dominance in geospatial ML, GDAL/rasterio, PyTorch. |
| API framework | FastAPI | Async, automatic OpenAPI docs, Pydantic schema validation, lightweight. |
| ASGI server (dev) | Uvicorn | Standard FastAPI dev server. |
| Image I/O | rasterio + GDAL | Native GeoTIFF/COG reading, CRS handling. |
| Geospatial ops | pyproj, shapely, geopandas | Reprojection, polygon/bbox ops, spatial indexing. |
| ML framework | PyTorch 2.x | Mamba-ssm requires PyTorch; also hosts CLIP-based models. |
| Mamba SSM | mamba-ssm (causal-conv1d) | Official implementation; requires CUDA for full speed (CPU fallback available). |
| Embedding model | RemoteCLIP (RS5M checkpoint) or GeoRSCLIP | RS-specific vision-language model for text↔image retrieval. |
| Change detection baseline | Pixel-difference + Otsu threshold | No-ML baseline; deterministic, fast. |
| Change detection learned | BIT-CD or ChangeFormer (pretrained on LEVIR-CD/WHU-CD) | Publicly available weights; works on Sentinel-2 patches. |
| Vector store | FAISS (IndexFlatIP + IVF) | Offline, no server required, numpy-compatible, fast ANN. |
| Metadata/spatial store | SQLite + SpatiaLite extension | Offline relational + spatial queries, no server. |
| Temporal state store | HDF5 (h5py) | Efficient storage of Mamba hidden state tensors per tile per timestamp. |
| Configuration | Pydantic-Settings + .env | Type-safe config, no secrets in code. |
| Task queue (Phase B+) | Celery + Redis or APScheduler | Deferred for Phase B; not needed in prototype. |

### 3.2 Frontend
| Component | Choice | Justification |
|-----------|--------|---------------|
| Framework | React 18 + Vite | Fast HMR, component model fits visualization modules. |
| Language | TypeScript | Type safety for API contracts; catches schema mismatches early. |
| Styling | Vanilla CSS (CSS Modules) | Per PROJECT_PLAN styling guideline; no Tailwind. |
| Map | Leaflet + react-leaflet | Offline-capable, no API key, well-documented. |
| Image viewer | OpenLayers or custom canvas overlay | Satellite imagery and change mask overlays. |
| State management | Zustand | Lightweight, no boilerplate, sufficient for single-user prototype. |
| API client | Axios + custom typed wrappers | Consistent error handling and request/response typing. |
| Charts | Recharts | Confidence/metadata charts; small bundle. |
| Testing | Vitest + React Testing Library | Compatible with Vite. |

### 3.3 Data Handling
| Component | Choice | Justification |
|-----------|--------|---------------|
| Language | Python 3.11 | Same runtime as backend; shared utilities. |
| Image I/O | rasterio, numpy, Pillow | Multi-band TIFF reading, normalization, format conversion. |
| Geospatial | pyproj, shapely | CRS normalization, tile ID generation. |
| Testing | pytest | Standard; fixtures for synthetic tiles. |

### 3.4 Shared Tooling
| Component | Choice |
|-----------|--------|
| Dependency management | `uv` (Python) / `npm` (Node) |
| Linting (Python) | Ruff |
| Formatting (Python) | Black + isort (via Ruff) |
| Type checking (Python) | mypy (strict mode on core interfaces) |
| Linting/formatting (TS) | ESLint + Prettier |
| Git hooks | pre-commit |
| Secrets management | `.env.local` (gitignored) + `.env.example` templates |
| Container (optional) | Docker Compose (deferred until Phase B) |
| CI (optional, Phase C) | GitHub Actions |

---

## 4. Repository Architecture and Complete Proposed Folder Tree

```
terraeyes/
├── .env.example                    # Config template (no secrets)
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── pyproject.toml                  # Root Python project (workspace root for uv)
├── uv.lock
│
├── setup/                          # Dev tooling, environment docs, shared conventions
│   ├── README.md                   # Onboarding and setup instructions
│   ├── requirements.txt            # Pinned production Python deps (generated from uv)
│   ├── requirements-dev.txt        # Dev/test extras
│   ├── model_registry.md           # Pretrained model catalogue (name, licence, URL, local path)
│   ├── branch_conventions.md
│   ├── coding_standards.md
│   └── docker-compose.yml          # Optional (Phase B+)
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── public/
│   │   └── assets/                 # Static icons, fonts
│   └── src/
│       ├── main.tsx                # App bootstrap
│       ├── App.tsx                 # Router root
│       ├── routes/                 # Page-level route components
│       │   ├── SearchPage.tsx
│       │   ├── ComparisonPage.tsx
│       │   └── ResultPage.tsx
│       ├── layout/
│       │   ├── AppShell.tsx        # Top nav + sidebar wrapper
│       │   ├── Sidebar.tsx
│       │   └── TopBar.tsx
│       ├── features/
│       │   ├── search/
│       │   │   ├── SearchBar.tsx
│       │   │   ├── QueryTypeToggle.tsx   # Text vs. image query
│       │   │   ├── FilterPanel.tsx       # Location, date range, sensor
│       │   │   ├── SearchStore.ts        # Zustand slice
│       │   │   ├── searchValidation.ts
│       │   │   └── search.types.ts
│       │   ├── results/
│       │   │   ├── ResultList.tsx
│       │   │   ├── ResultCard.tsx
│       │   │   ├── ConfidenceBadge.tsx
│       │   │   ├── MetadataPanel.tsx
│       │   │   └── results.types.ts
│       │   ├── comparison/
│       │   │   ├── ComparisonViewer.tsx  # Before/after image panel
│       │   │   ├── DateSelector.tsx
│       │   │   ├── SliderSplit.tsx       # Swipe/split comparison
│       │   │   └── comparison.types.ts
│       │   ├── visualization/
│       │   │   ├── ChangeMaskOverlay.tsx
│       │   │   ├── BoundingBoxLayer.tsx
│       │   │   ├── MapView.tsx           # Leaflet wrapper
│       │   │   ├── ImageTileRenderer.tsx
│       │   │   └── visualization.types.ts
│       │   └── summary/
│       │       ├── ResultSummaryPanel.tsx
│       │       ├── ChangeTimeline.tsx
│       │       └── summary.types.ts
│       ├── api/
│       │   ├── client.ts           # Axios instance, base URL, interceptors
│       │   ├── health.api.ts
│       │   ├── search.api.ts
│       │   ├── images.api.ts
│       │   ├── comparison.api.ts
│       │   └── changeDetection.api.ts
│       ├── store/
│       │   ├── index.ts            # Combine Zustand slices
│       │   ├── appStore.ts         # Global UI state
│       │   └── sessionStore.ts     # Active query session state
│       ├── components/             # Shared UI primitives
│       │   ├── Button.tsx
│       │   ├── Spinner.tsx
│       │   ├── ErrorBanner.tsx
│       │   ├── EmptyState.tsx
│       │   └── Modal.tsx
│       ├── types/
│       │   ├── api.types.ts        # Mirror of backend response schemas
│       │   └── geo.types.ts        # Coordinate, bbox, CRS types
│       ├── styles/
│       │   ├── global.css
│       │   ├── variables.css       # Design tokens
│       │   └── typography.css
│       └── tests/
│           ├── components/
│           └── api/
│
├── backend/
│   ├── pyproject.toml
│   ├── main.py                     # FastAPI app factory + startup
│   ├── config.py                   # Pydantic-Settings config object
│   ├── logging_config.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── router.py           # Mount all versioned routers
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── health.py
│   │   │       ├── search.py
│   │   │       ├── images.py
│   │   │       ├── comparison.py
│   │   │       └── change_detection.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── search.py           # Request/response Pydantic models
│   │   │   ├── images.py
│   │   │   ├── comparison.py
│   │   │   ├── change_detection.py
│   │   │   └── common.py           # BBox, Coordinate, Confidence, Timestamp
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── query_orchestrator.py  # Routes query to retrieval + detection
│   │   │   ├── retrieval_service.py   # Calls embedding model + vector store
│   │   │   ├── image_service.py       # Fetches tile bytes + metadata from data layer
│   │   │   ├── comparison_service.py  # Selects bi-temporal image pairs
│   │   │   ├── change_detection_service.py  # Calls model interface
│   │   │   ├── summary_service.py     # Assembles final structured result
│   │   │   └── clustering_service.py  # Phase B
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py       # ABCs: EmbeddingModel, ChangeDetector
│   │   │   ├── embedding/
│   │   │   │   ├── remote_clip.py  # RemoteCLIP adapter
│   │   │   │   └── mock_embedding.py  # Prototype mock
│   │   │   └── change_detection/
│   │   │       ├── pixel_diff.py   # Baseline pixel-difference detector
│   │   │       ├── bit_cd.py       # BIT-CD adapter (Phase B)
│   │   │       └── mamba_cd.py     # Mamba-based detector (Phase C)
│   │   ├── vector_store/
│   │   │   ├── __init__.py
│   │   │   ├── faiss_store.py      # FAISS index wrapper
│   │   │   └── store_interface.py  # ABC for vector stores
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── connection.py       # SQLite + SpatiaLite setup
│   │   │   ├── models.py           # SQLAlchemy ORM models
│   │   │   ├── migrations/         # Alembic migration scripts
│   │   │   └── repositories/
│   │   │       ├── tile_repo.py
│   │   │       ├── metadata_repo.py
│   │   │       └── temporal_state_repo.py
│   │   ├── geospatial/
│   │   │   ├── __init__.py
│   │   │   ├── crs.py              # CRS normalization, reprojection
│   │   │   ├── tiling.py           # Tile ID generation, bbox math
│   │   │   └── alignment.py        # Co-registration helpers
│   │   ├── postprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── mask_refinement.py  # Morphological ops on change mask
│   │   │   ├── bbox_extraction.py  # Connected components → bounding boxes
│   │   │   └── confidence.py       # Confidence calibration utilities
│   │   └── exceptions.py
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── conftest.py
│
├── data-handling/
│   ├── pyproject.toml
│   ├── README.md
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── dataset_adapter.py      # ABC: DatasetAdapter
│   │   └── tile_provider.py        # ABC: TileProvider
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── oscd/
│   │   │   ├── __init__.py
│   │   │   ├── oscd_adapter.py     # DatasetAdapter impl for OSCD
│   │   │   ├── oscd_loader.py      # File discovery, band reading
│   │   │   ├── oscd_metadata.py    # Metadata extraction/normalization
│   │   │   └── oscd_schema.md      # Expected directory layout (to verify on download)
│   │   ├── sentinel2/              # Phase B
│   │   ├── sentinel1/              # Phase B
│   │   └── landsat/                # Phase B
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── band_selection.py       # Band indices, normalization
   │   ├── cloud_masking.py        # Optical sensor cloud/shadow masks
│   │   ├── speckle_filter.py       # SAR speckle reduction
│   │   ├── coregistration.py       # ECC / phase correlation
│   │   ├── tiling.py               # Split scene into tiles with stable IDs
│   │   ├── normalization.py        # Per-band statistics normalization
│   │   └── pipeline.py             # Composable preprocessing pipeline
│   ├── geospatial/
│   │   ├── crs_utils.py
│   │   ├── tile_id.py              # Geographic tile ID scheme
│   │   └── alignment_check.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── ingest_oscd.py          # CLI: ingest OSCD into the system stores
│   │   └── incremental_ingest.py   # Add new scenes without full rebuild
│   ├── fixtures/
│   │   ├── synthetic_tiles/        # Small numpy arrays for unit tests
│   │   └── mock_metadata.json
│   ├── splits/
│   │   └── oscd_splits.json        # Train/val/test city splits
│   └── tests/
│       ├── test_oscd_loader.py
│       ├── test_preprocessing.py
│       └── conftest.py
│
├── docs/
│   ├── architecture.md
│   ├── api_reference.md            # Auto-generated from OpenAPI
│   ├── model_registry.md
│   ├── data_provenance.md
│   └── adr/                        # Architecture Decision Records
│       ├── ADR-001-faiss-vs-chroma.md
│       └── ADR-002-mamba-vs-transformer.md
│
├── tests/
│   └── e2e/                        # End-to-end tests (Playwright or httpx)
│
└── scripts/
    ├── download_models.py          # Fetch pretrained weights (run once at setup)
    ├── verify_dataset.py           # Check OSCD directory after download
    ├── build_index.py              # Build FAISS index from embeddings
    └── evaluate.py                 # Retrieval + change-detection metrics
```

---

## 5. Responsibility of Every Major Folder and Module

### 5.1 `setup/`
Owns all one-time project-wide configuration decisions. No application logic. Consumed by all other areas. Should be the first PR merged.

### 5.2 `frontend/src/features/search/`
Owns the query input UX: text vs. image query toggle, filter panel (location, date range, sensor), form validation, and submission to the API. Depends on `api/search.api.ts` only.

### 5.3 `frontend/src/features/results/`
Owns the ranked result list, individual result cards, confidence badge display, and metadata panel. Depends on `api/search.api.ts` response type.

### 5.4 `frontend/src/features/comparison/`
Owns the bi-temporal image viewer: date selector, split/swipe comparison widget. Consumes image URLs from `api/images.api.ts` and `api/comparison.api.ts`.

### 5.5 `frontend/src/features/visualization/`
Owns map rendering (Leaflet), change mask overlay, bounding box layer rendering on canvas/SVG, and tile image rendering. Consumes image bytes and mask data from the backend.

### 5.6 `frontend/src/features/summary/`
Owns the structured result summary panel: location, time range, detected change type, earliest supported change date, confidence, source provenance, and change timeline chart.

### 5.7 `frontend/src/api/`
The only layer that makes HTTP calls. Every API module exports typed async functions that match the backend OpenAPI schema exactly. Frontend feature modules must not call `axios` directly.

### 5.8 `backend/app/api/v1/`
Thin routing layer. No business logic. Validates requests via Pydantic schemas, delegates to services, serializes responses. Owns HTTP status codes and error response structure.

### 5.9 `backend/app/services/`
Business logic and orchestration. Model-agnostic: services call `interfaces.EmbeddingModel` and `interfaces.ChangeDetector`, not concrete implementations. No direct FAISS or rasterio calls here.

### 5.10 `backend/app/models/interfaces.py`
The critical abstraction boundary. Defines `EmbeddingModel` and `ChangeDetector` ABCs. Any swap of underlying ML model must not require changes outside `models/`. Backend services are permanently stable against this interface.

### 5.11 `backend/app/models/embedding/`
Concrete embedding model adapters. `mock_embedding.py` is used in the prototype and returns random unit vectors of fixed dimension. `remote_clip.py` wraps the RemoteCLIP model after Phase A approval.

### 5.12 `backend/app/models/change_detection/`
`pixel_diff.py` is the prototype baseline: compute per-pixel absolute difference, threshold with Otsu's method, return binary mask. `bit_cd.py` and `mamba_cd.py` are Phase B/C implementations.

### 5.13 `backend/app/vector_store/`
FAISS wrapper. Exposes `add(embeddings, ids)`, `search(query_embedding, k)`, `save()`, `load()`. Can be swapped for ChromaDB or Qdrant without touching services.

### 5.14 `backend/app/db/`
SQLite + SpatiaLite for tile metadata, geospatial bounding boxes, ingestion provenance, and temporal state references. ORM models and Alembic migrations here.

### 5.15 `backend/app/geospatial/`
Pure coordinate math: reprojection, tile bbox calculation, tile ID logic. No I/O.

### 5.16 `backend/app/postprocessing/`
Takes raw change mask tensors, applies morphological cleaning, extracts connected-component bounding boxes, and computes confidence scores. No ML model calls.

### 5.17 `data-handling/interfaces/`
`DatasetAdapter` ABC: exposes `list_locations()`, `list_dates(location_id)`, `load_tile(location_id, date, tile_id, bands)`, `load_mask(location_id, date_pair)`. The backend never imports from `adapters/` directly; it imports from `interfaces/`.

### 5.18 `data-handling/adapters/oscd/`
OSCD-specific file discovery, band reading (rasterio), metadata normalization. Implements `DatasetAdapter`. To be finalized when OSCD is downloaded and directory structure is verified.

### 5.19 `data-handling/preprocessing/pipeline.py`
Composable, ordered preprocessing pipeline. Each step is a callable that takes and returns a normalized data record. Steps are configured per sensor type via a dict/YAML config, not hardcoded.

### 5.20 `data-handling/fixtures/`
Synthetic 256×256 numpy tile arrays in known bands with known change masks. Used exclusively by unit tests. Never used in production.

---

## 6. Foundational Setup Plan

### 6.1 Tasks That Must Complete Before Parallel Development Begins

These are blocking tasks; parallel workstreams cannot start until they are done:

| # | Task | Owner | Output |
|---|------|-------|--------|
| S1 | Initialize Git repository; establish `main`, `dev`, `feature/*` branch conventions. | WS-1 | `.gitignore`, `README.md`, branch protection rules |
| S2 | Create root `pyproject.toml` with Python 3.11 constraint and `uv` workspace; add `backend/` and `data-handling/` as sub-packages. | WS-1 | Installable Python workspace |
| S3 | Create `frontend/` Vite + React + TypeScript scaffold using `npx create-vite@latest ./ --template react-ts`. | WS-1 | Running `npm run dev` |
| S4 | Define `.env.example` with all required environment variables. | WS-1 | Template with documented keys |
| S5 | Install and configure `pre-commit` with Ruff, Black, mypy, ESLint, Prettier hooks. | WS-1 | Hooks enforced on commit |
| S6 | Publish **API contract v0** (Section 12 of this plan) in `docs/api_reference.md`. | WS-1 | Agreed request/response schemas |
| S7 | Implement `backend/app/models/interfaces.py` (ABCs only, no implementations). | WS-1 | Importable interface module |
| S8 | Implement `data-handling/interfaces/` ABCs. | WS-1 | Importable interface module |
| S9 | Create `data-handling/fixtures/` with 2 synthetic 256×256 tiles (before/after, with known binary change mask). | WS-1 | pytest-loadable fixtures |
| S10 | Configure pytest for `backend/tests/` and `data-handling/tests/`. | WS-1 | `pytest` passes with 0 real tests yet |

### 6.2 Tasks That Can Be Deferred

| Task | Deferred Until |
|------|---------------|
| Docker Compose | Phase B |
| Alembic migrations | After SQLite schema is stable (Phase A end) |
| GitHub Actions CI | Phase B |
| Model weight download script | Before Phase B model integration |
| Redis / Celery task queue | Phase B+ |

### 6.3 Environment Variables (`.env.example`)

```env
# Backend
TERRAEYES_ENV=development
TERRAEYES_HOST=0.0.0.0
TERRAEYES_PORT=8000
TERRAEYES_LOG_LEVEL=INFO
TERRAEYES_DB_PATH=./data/terraeyes.db
TERRAEYES_FAISS_INDEX_PATH=./data/faiss.index
TERRAEYES_TEMPORAL_STORE_PATH=./data/temporal_states.h5
TERRAEYES_MODELS_DIR=./models
TERRAEYES_DATA_DIR=./data
TERRAEYES_OSCD_DIR=./data/oscd          # Path to downloaded OSCD (blank until downloaded)
TERRAEYES_EMBEDDING_MODEL=mock          # mock | remote_clip | georscclip
TERRAEYES_CHANGE_DETECTOR=pixel_diff    # pixel_diff | bit_cd | mamba_cd
TERRAEYES_TILE_SIZE=256
TERRAEYES_EMBEDDING_DIM=512

# Frontend
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

### 6.4 Logging Convention

- Python: `structlog` with JSON output in production, colored console in development.
- Log levels: `DEBUG` for model internals, `INFO` for request/response lifecycle, `WARNING` for degraded fallback, `ERROR` for handled exceptions, `CRITICAL` for startup failures.
- Every log entry must include: `request_id`, `module`, `timestamp`.
- Frontend: `console.error` only for unhandled exceptions; no `console.log` in production builds.

### 6.5 Branch and PR Conventions

- `main`: protected; CI must pass; only squash-merge from `dev`.
- `dev`: integration branch; direct pushes allowed by WS-1 during setup only.
- `feature/<workstream>/<short-description>`: feature branches; rebase onto `dev`.
- PR template must reference the relevant workstream, acceptance criteria, and test results.

---

## 7. Foundational Working Prototype Plan (Phase A)

### 7.1 Prototype Scope and Non-Scope

**In scope (Phase A):**
- React frontend with search bar, filter panel, result list, before/after comparison view, change mask overlay, and summary panel.
- FastAPI backend with health, search, images, comparison, and change-detection endpoints (all v1).
- Mock embedding model (returns random unit vectors).
- Pixel-difference baseline change detector (real computation, no ML).
- 3–5 synthetic/sample tile pairs baked into fixtures (small JPEG or PNG crops from a public source, e.g., EuroSAT or ESA open data, not OSCD).
- SQLite metadata store with the tiles pre-seeded.
- FAISS flat index pre-built from mock embeddings for the sample tiles.
- End-to-end flow: user submits text query → backend returns ranked results → user selects a result → comparison view loads before/after images → change detection runs → change mask and summary panel rendered.

**Out of scope (Phase A):**
- Real embedding model (RemoteCLIP).
- OSCD integration.
- Mamba-based models.
- Landsat/Sentinel-1/Sentinel-2 real data.
- Spatial index queries.
- Clustering.
- Analyst feedback/auditing.
- Incremental ingestion.

### 7.2 Sample Imagery Strategy (Phase A)

Use **3 city snapshots** from publicly available open data (e.g., EuroSAT RGB crops, Copernicus Open Access Hub previews, or NASA Earthdata preview images) under open licences. These are pre-downloaded and committed as small PNG files (< 500 KB each) in `data-handling/fixtures/synthetic_tiles/`. Each pair consists of a "before" and "after" image with a hand-crafted binary change mask PNG.

Document licence and origin of each sample image in `docs/data_provenance.md`.

### 7.3 Prototype Milestone Checklist

| # | Milestone | Acceptance Criterion |
|---|-----------|----------------------|
| PA-1 | Backend starts and `/health` returns 200. | `curl http://localhost:8000/api/v1/health` → `{"status":"ok"}` |
| PA-2 | Backend `POST /search` returns ranked results from mock data. | 3 results with score, location, dates, sensor. |
| PA-3 | Backend `GET /images/{tile_id}/{date}` returns image bytes. | Browser can render the PNG. |
| PA-4 | Backend `POST /comparison` returns before/after image URLs. | Both URLs resolve to valid images. |
| PA-5 | Backend `POST /change-detection` returns binary mask + bounding boxes + confidence. | Mask is a valid PNG; bboxes are non-empty. |
| PA-6 | Frontend renders search bar and submits query. | Network tab shows `POST /search` request. |
| PA-7 | Frontend renders result list with 3 cards. | Each card shows location, date, sensor, confidence. |
| PA-8 | Frontend comparison view shows before/after images side by side. | Both images visible, no 404. |
| PA-9 | Frontend renders change mask as semi-transparent overlay on "after" image. | Red/orange pixels visible where change mask is 1. |
| PA-10 | Frontend summary panel shows location, date range, change type, confidence. | Correct values from backend response. |
| PA-11 | All prototype API tests pass. | `pytest backend/tests/` exits 0. |
| PA-12 | All frontend component tests pass. | `npx vitest run` exits 0. |

---

## 8. Frontend Implementation Plan

### 8.1 Module: Application Bootstrap and Routing (`src/main.tsx`, `src/App.tsx`, `src/routes/`)

**Responsibilities:** Mount React app, configure React Router v6, lazy-load page components.

**Routes:**
- `/` → `SearchPage` (default landing)
- `/results` → `ResultPage` (receives query state from store)
- `/compare/:locationId` → `ComparisonPage`

**Acceptance criterion:** All routes render without errors; React Router handles unknown paths with a 404 component.

### 8.2 Module: Shared Layout (`src/layout/`)

**Responsibilities:** `AppShell` wraps every page with `TopBar` (logo, nav links, theme toggle) and optional `Sidebar`. No business logic; purely structural.

**Acceptance criterion:** Layout renders consistently on all routes; sidebar collapses to icon-only on narrow viewports.

### 8.3 Module: Search and Query Input (`src/features/search/`)

**Files:** `SearchBar.tsx`, `QueryTypeToggle.tsx`, `FilterPanel.tsx`, `SearchStore.ts`, `searchValidation.ts`, `search.types.ts`

**Inputs:** User keyboard input, image file upload (image query mode), filter selections.

**Outputs:** Validated `SearchRequest` object dispatched to `search.api.ts`.

**State (Zustand slice):** `queryText`, `queryType`, `imageFile`, `filters: {location, dateFrom, dateTo, sensor}`, `isLoading`, `error`.

**Validation rules:**
- Text query: 3–500 characters.
- Date range: `dateFrom` ≤ `dateTo`; not in the future; date format ISO 8601.
- Sensor: one of `["landsat", "sentinel-1", "sentinel-2", "any"]`.

**Acceptance criterion:** Invalid submissions show inline error messages; valid submission triggers `POST /search`.

### 8.4 Module: Results (`src/features/results/`)

**Files:** `ResultList.tsx`, `ResultCard.tsx`, `ConfidenceBadge.tsx`, `MetadataPanel.tsx`, `results.types.ts`

**Inputs:** `SearchResponse` array from `SearchStore`.

**Displays per card:** Thumbnail, location name, date range, sensor, change type (if available), confidence score as coloured badge.

**Acceptance criterion:** List renders 0, 1, or many results gracefully; clicking a card navigates to `/compare/:locationId`.

### 8.5 Module: Temporal Comparison (`src/features/comparison/`)

**Files:** `ComparisonViewer.tsx`, `DateSelector.tsx`, `SliderSplit.tsx`, `comparison.types.ts`

**Inputs:** `locationId` from route params, available dates from `GET /images/:locationId/dates`.

**Behaviour:** User selects two dates; images are fetched and displayed in a synchronized split/swipe viewer. Images are pixel-aligned before display.

**Acceptance criterion:** Slider transitions smoothly; both images load within 3 s on local network; selecting new dates re-fetches correctly.

### 8.6 Module: Visualization (`src/features/visualization/`)

**Files:** `ChangeMaskOverlay.tsx`, `BoundingBoxLayer.tsx`, `MapView.tsx`, `ImageTileRenderer.tsx`, `visualization.types.ts`

**Inputs:** Image tile bytes (PNG/JPEG), change mask bytes (PNG, binary), bounding box array `[{x,y,w,h,label,confidence}]`, geographic bbox for map positioning.

**Behaviour:** `ChangeMaskOverlay` renders mask as a CSS `mix-blend-mode: multiply` semi-transparent red layer over the "after" image. `BoundingBoxLayer` renders SVG rectangles with confidence labels. `MapView` shows a Leaflet map centred on the tile geographic bbox.

**Acceptance criterion:** Mask and bboxes are correctly aligned to the image; toggling overlay visibility works without re-fetching.

### 8.7 Module: Result Summary (`src/features/summary/`)

**Files:** `ResultSummaryPanel.tsx`, `ChangeTimeline.tsx`, `summary.types.ts`

**Inputs:** `SummaryResponse` from `POST /change-detection`.

**Displays:** Location, date range, detected change type, earliest detectable change date, confidence score, source provenance string, optional change timeline chart.

**Acceptance criterion:** All fields render correctly from backend response; missing optional fields show placeholder text, not undefined/null.

### 8.8 Module: API Client (`src/api/`)

**Rules:**
- Single Axios instance in `client.ts` with `baseURL = import.meta.env.VITE_API_BASE_URL`.
- All errors caught and re-thrown as typed `ApiError` objects.
- Request/response types in `src/types/api.types.ts` must mirror backend Pydantic schemas.
- No API call code outside `src/api/`.

### 8.9 Frontend Tests

- **Component tests:** Every `features/` component has a corresponding test file. Use `@testing-library/react`.
- **API mock tests:** `src/api/` modules tested with MSW (Mock Service Worker) interceptors.
- **Coverage target (prototype):** ≥ 70% branch coverage for `features/` and `api/` modules.

---

## 9. Backend Implementation Plan

### 9.1 Architecture Decision: Modular Monolith

The backend is a **modular monolith**, not microservices. Justification:
- Single-user local operation in Phase A/B; no network partitioning needed.
- Avoids inter-service serialization overhead for large image tensors.
- Team can refactor service boundaries later if scale requires it.
- FastAPI's dependency injection is sufficient for decoupling.

Revisit this decision at Phase C if concurrent multi-user or large-scale ingestion is required.

### 9.2 Module: Application Initialization (`main.py`, `config.py`)

`main.py` uses `lifespan` context manager to:
1. Load and validate config from `.env`.
2. Initialize SQLite DB (create tables if not exist).
3. Load FAISS index from disk (or create empty index if not found).
4. Instantiate embedding model and change detector as singletons (from config).
5. Register all v1 routers.

**Acceptance criterion:** `uvicorn main:app --reload` starts without errors; `/health` returns 200.

### 9.3 Module: API Routing (`app/api/v1/`)

Each endpoint file is thin:
```python
@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, orchestrator: QueryOrchestrator = Depends(get_orchestrator)):
    return await orchestrator.search(req)
```

All business logic is in `services/`. Routers own only HTTP concerns.

### 9.4 Module: Services Layer (`app/services/`)

**`query_orchestrator.py`:** Entry point for search requests. Calls `retrieval_service` for embedding-based results, merges with metadata filter results, returns ranked list.

**`retrieval_service.py`:** Calls `EmbeddingModel.encode_text(query)` → vector → `VectorStore.search(vector, k)` → tile IDs → metadata lookup from `tile_repo`.

**`image_service.py`:** Given `(tile_id, date)`, reads the tile image from filesystem (via `DatasetAdapter`), returns bytes + metadata dict.

**`comparison_service.py`:** Given `(location_id, date1, date2)`, calls `image_service` twice, validates both tiles exist and are co-registered.

**`change_detection_service.py`:** Calls `ChangeDetector.detect(before_array, after_array)` → binary mask → `mask_refinement.py` → `bbox_extraction.py` → `confidence.py` → assembles `ChangeDetectionResult`.

**`summary_service.py`:** Assembles `SummaryResponse` from retrieval results, change detection results, and metadata.

### 9.5 Module: Model Interfaces (`app/models/interfaces.py`)

```python
class EmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray: ...
    @abstractmethod
    def encode_image(self, image: np.ndarray) -> np.ndarray: ...
    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

class ChangeDetector(ABC):
    @abstractmethod
    def detect(self, before: np.ndarray, after: np.ndarray) -> ChangeDetectionOutput: ...
```

`ChangeDetectionOutput` is a dataclass: `mask: np.ndarray, confidence: float, metadata: dict`.

**Rule:** No concrete model class may be imported anywhere except `app/models/{embedding,change_detection}/` and `main.py` factory.

### 9.6 Module: Prototype Mock Models

`mock_embedding.py`: `encode_text` → `np.random.randn(512).astype(np.float32)` (normalized). `encode_image` → same. Deterministic seed per text hash so queries return stable results during development.

`pixel_diff.py`: Compute per-channel absolute difference. Convert to single-channel by max-projection. Apply Otsu threshold (skimage). Return binary mask. Confidence = 1 - (fraction of unchanged pixels).

### 9.7 Module: Vector Store (`app/vector_store/faiss_store.py`)

- Wraps `faiss.IndexFlatIP` (inner product, i.e., cosine on normalized vectors).
- Methods: `add(embeddings: np.ndarray, ids: list[str])`, `search(query: np.ndarray, k: int) -> list[tuple[str,float]]`, `save()`, `load()`.
- Thread-safe via `threading.Lock` for Phase A single-user prototype; revisit for Phase B.

### 9.8 Module: Database (`app/db/`)

Initial SQLite schema (to be created via Alembic):
```sql
tiles (tile_id TEXT PK, location_id TEXT, sensor TEXT, date TEXT, bbox_json TEXT,
       crs TEXT, resolution_m REAL, embedding_id TEXT, ingested_at TEXT)
metadata (tile_id TEXT, key TEXT, value TEXT)
temporal_states (tile_id TEXT, date TEXT, state_path TEXT, updated_at TEXT)
```
SpatiaLite geometry column added in Phase B for spatial queries.

### 9.9 Module: Postprocessing (`app/postprocessing/`)

**`mask_refinement.py`:** Remove small noise regions (morphological opening, minimum area threshold configurable via env var). Fill small holes (morphological closing).

**`bbox_extraction.py`:** `skimage.measure.label` → `regionprops` → filter by area → return list of `BoundingBox(x, y, w, h, area, label="change")`.

**`confidence.py`:** Phase A: `changed_pixel_fraction * mask_quality_score` (heuristic). Phase B: calibrate against OSCD ground truth.

### 9.10 Backend Tests

- **Unit tests:** Every service method tested with mocked dependencies (pytest + unittest.mock).
- **API tests:** `httpx.AsyncClient` against the live app (in-process, no real network). Validate request/response schemas using Pydantic directly.
- **Contract tests:** For every endpoint, assert that the response schema exactly matches `src/types/api.types.ts` definitions (maintained manually in Phase A; consider openapi-typescript in Phase B).
- **Coverage target:** ≥ 80% for `services/` and `models/`.

---

## 10. Data-Handling and OSCD Integration Plan

### 10.1 Interface-First Design

The backend imports `DatasetAdapter` from `data-handling/interfaces/`. The OSCD implementation is a plugin; the backend never imports from `adapters/oscd/` directly.

### 10.2 Phase A (Pre-OSCD): Fixture-Based Provider

`FixtureTileProvider` in `data-handling/adapters/fixtures/fixture_adapter.py` implements `DatasetAdapter` using the synthetic PNG tiles in `fixtures/synthetic_tiles/`. This unblocks backend and frontend development before OSCD is downloaded.

### 10.3 OSCD Integration (Phase B)

**Step 1 — Verification (run `scripts/verify_dataset.py` after download):**
- Confirm directory structure matches `adapters/oscd/oscd_schema.md`.
- List all city directories and validate band TIF counts.
- Confirm CRS (expected: UTM or WGS84) and resolution.
- Confirm ground-truth mask availability and format (binary GeoTIFF).
- Report any missing or corrupted files.

**Step 2 — OSCD Loader (`oscd_loader.py`):**
- Discover `<city>/<date>/` subdirectories.
- For each city-date pair, read selected bands (e.g., RGB = B04, B03, B02; false-colour = B08, B04, B03).
- Return `np.ndarray` of shape `(H, W, C)` normalized to `[0, 1]`.

**Step 3 — Metadata Normalization (`oscd_metadata.py`):**
- Extract acquisition date, sensor (Sentinel-2 for OSCD), geographic bbox (from GeoTIFF transform), tile size.
- Normalize to the common `TileMetadata` schema used by the backend.

**Step 4 — Tiling (`preprocessing/tiling.py`):**
- Slice scenes into 256×256 (or configured) tiles.
- Assign stable tile IDs: `{location_id}_{row:04d}_{col:04d}_{sensor}`.
- Store tile bboxes in SQLite.

**Step 5 — Ingestion CLI (`ingestion/ingest_oscd.py`):**
- `python -m data_handling.ingestion.ingest_oscd --oscd-dir $TERRAEYES_OSCD_DIR --output-dir $TERRAEYES_DATA_DIR`
- Reads tiles → runs preprocessing pipeline → generates embeddings (via backend embedding model API) → writes to FAISS index and SQLite.
- Idempotent: skip tiles already in the index.

**Step 6 — Train/Val/Test Splits (`splits/oscd_splits.json`):**
- OSCD has 24 city pairs. Proposed split: 14 train / 5 val / 5 test.
- Test cities must never appear in the retrieval index during evaluation.

### 10.4 Preprocessing Pipeline Steps (per sensor)

| Step | Landsat | Sentinel-1 | Sentinel-2 |
|------|---------|------------|------------|
| Radiometric calibration | DN → TOA reflectance | σ⁰ calibration | DN → BOA reflectance (if L2A) |
| Cloud/shadow masking | Fmask or QA_PIXEL band | N/A | SCL band (L2A) or Sen2Cor |
| Speckle filter | N/A | Lee or Refined Lee | N/A |
| Terrain correction | N/A | Range Doppler (SNAP) | N/A |
| Co-registration | ECC algorithm vs. reference tile | same | same |
| Band normalization | Per-band z-score (channel statistics) | dB normalization | same as Landsat |
| Tiling | 256×256 with 32px overlap | same | same |

Note: Full preprocessing for Sentinel-1 (terrain correction) requires SNAP or equivalent. This is a Phase B prerequisite.

### 10.5 Sensor Support Timeline

| Sensor | Phase |
|--------|-------|
| OSCD (Sentinel-2 subset, limited preprocessing) | Phase B |
| Sentinel-2 (full pipeline) | Phase B |
| Landsat | Phase B |
| Sentinel-1 (SAR) | Phase C (requires SNAP + additional model validation) |

---

## 11. Model Candidates and Model-Selection Checkpoints

### 11.1 Semantic Retrieval Candidates

| Model | RS-specific? | Embedding dim | Inference (CPU) | Licence | Offline? |
|-------|-------------|---------------|-----------------|---------|----------|
| **RemoteCLIP** (RS5M-ViT-L-14) | ✅ Yes | 768 | ~2 s/img | MIT | ✅ Yes (download weights once) |
| **GeoRSCLIP** (ViT-H-14) | ✅ Yes | 1024 | ~4 s/img | Apache 2.0 | ✅ Yes |
| OpenCLIP (ViT-L-14, general) | ❌ No | 768 | ~2 s/img | MIT | ✅ Yes |
| CLIP (OpenAI ViT-B-32) | ❌ No | 512 | ~0.5 s/img | MIT | ✅ Yes |

**Recommendation for Checkpoint CP-1:** Start with CLIP (ViT-B-32) as the Phase B integration test due to smallest size. Evaluate RemoteCLIP recall@5 on a held-out OSCD subset before committing.

---
### 🚦 Model-Selection Checkpoint CP-1: Semantic Retrieval Model

**Decision:** Which embedding model to use for text→image retrieval.

**Alternatives:**
1. CLIP (ViT-B-32) — smallest, fastest, not RS-specific.
2. RemoteCLIP (RS5M-ViT-L-14) — RS-specific, larger, slower.
3. GeoRSCLIP (ViT-H-14) — largest, most powerful, most memory.

**Evidence needed:** Run retrieval on 5 held-out OSCD city pairs. Measure Recall@5, Recall@10, and mean reciprocal rank for 10 hand-crafted text queries (e.g., "urban expansion near river", "deforestation").

**Costs:** RemoteCLIP requires ~3 GB VRAM or ~8 GB RAM for CPU inference. GeoRSCLIP requires ~6 GB VRAM.

**Stop condition:** Present comparison table to user. Await approval before loading model weights into backend.

---

### 11.2 Change Detection Candidates

| Method | Type | Supervision | Notes |
|--------|------|-------------|-------|
| **Pixel difference + Otsu** | Baseline | None | Fast, no weights, high false positives |
| **CVA (Change Vector Analysis)** | Conventional | None | Better than diff; sensitive to registration |
| **BIT-CD** (Bitemporal Image Transformer) | Learned | LEVIR-CD + WHU-CD | Strong on building changes; pretrained weights public |
| **ChangeFormer** | Learned | LEVIR-CD | Transformer-based; strong results on optical |
| **Mamba-based CD (custom)** | Learned (Mamba SSM) | Requires training on OSCD | Per reference architecture; requires GPU training |

**Phase A:** Pixel difference + Otsu (real implementation, no ML weights needed).

**Phase B:** Evaluate BIT-CD or ChangeFormer on OSCD after dataset integration.

---
### 🚦 Model-Selection Checkpoint CP-2: Change Detection Model

**Decision:** Which change detector to deploy after the baseline.

**Alternatives:**
1. CVA — no weights needed, marginal improvement over pixel diff.
2. BIT-CD — pretrained weights, strong building change detection, ~50 MB.
3. ChangeFormer — pretrained weights, Transformer-based, ~120 MB.
4. Mamba-based (custom) — requires training from scratch on OSCD; highest potential but highest cost.

**Evidence needed:** Evaluate F1-score and IoU on 5 OSCD test cities for BIT-CD and ChangeFormer vs. baseline. Compare inference speed.

**Stop condition:** Present evaluation results to user. Await approval before replacing pixel-diff baseline.

---

### 11.3 Temporal Mamba Architecture (Reference Image: Spatio-Feature Extraction, Temporal Feature Extraction, Change Detector, Semantic Change Detection)

The reference architecture shows four Mamba-based modules:

| Module | Role | Input | Output |
|--------|------|-------|--------|
| Spatio-Feature Extraction | Extract spatial features per tile | Dense embeddings | Feature map + Spatio DB update |
| Temporal Feature Extraction | Update temporal hidden state per location | Dense embeddings + previous Mamba state | Updated temporal state |
| Change Detector | Detect if change occurred | Dense embeddings + temporal states | Binary decision + rough mask |
| Semantic Change Detection | Classify type and extent of change | Dense embeddings + change signal | Semantic change mask |

This is a **Phase C** architecture. It requires:
- Mamba SSM implementation (`mamba-ssm` package, CUDA required for practical training).
- Training on OSCD or similar dataset with temporal pairs.
- Temporal state persistence across inference calls (HDF5 store).
- State update logic: temporal state must be updated after every Change Detector run, even if no change is detected (per the reference diagram note).

---
### 🚦 Model-Selection Checkpoint CP-3: Mamba Architecture Design

**Decision:** Custom Mamba architecture design (dimensions, depth, state size) and training strategy.

**Stop condition:** Present proposed architecture, training data requirements, and compute estimate. Await approval before beginning any training code or Mamba integration.

---

### 11.4 Satellite-Specific Model Compatibility

| Concern | Impact |
|---------|--------|
| Landsat: 6–7 optical bands, 30 m resolution | CLIP/RemoteCLIP trained on RGB; band selection and RGB composite required. |
| Sentinel-2: 13 bands, 10/20/60 m mixed | RGB or false-colour composite for optical models; full band stack for specialized models. |
| Sentinel-1: SAR, 2 polarizations (VV, VH), backscatter values | Optical models cannot process directly; requires SAR-specific model or dual-pol RGB visualization. |
| Cross-sensor change detection | Optical↔SAR comparison not directly supported by pixel diff or optical CD models; requires domain adaptation or separate modality-specific pipelines. |

**Rule:** Never pass raw Sentinel-1 backscatter to an optical embedding model. Always validate sensor type before model dispatch.

---

### 11.5 Remote Training Infrastructure (Google Colab / Kaggle)

As per project constraints, local GPU training is avoided due to hardware limitations, but cost must remain zero since OSCD is a relatively small dataset. 
Therefore, **Google Colab (or Kaggle Notebooks)** is designated as the remote training infrastructure.

**Workflow:**
1. **Local Development:** Data loading scripts, `DatasetAdapter`, and model architectures (PyTorch/Mamba) are developed and version-controlled locally.
2. **Notebook Export:** A Jupyter Notebook (`training_pipeline.ipynb`) is maintained in `notebooks/`. It clones the repository or installs the local package in the Colab environment.
3. **Remote Execution:** The notebook downloads the OSCD dataset directly into the ephemeral Colab environment, runs the training loop using the free T4/L4 GPU, and exports the final `.pt` or `.onnx` weights.
4. **Local Integration:** The trained weights are downloaded locally into `model_registry/` for inference within the local API.

**Evidence needed:** Ensure `training_pipeline.ipynb` executes cleanly top-to-bottom on a free Colab instance without OOM errors.

---

## 12. API Contracts and Shared Schemas

### 12.1 Common Types

```typescript
// src/types/api.types.ts (TypeScript mirror of Pydantic schemas)

type Sensor = "landsat" | "sentinel-1" | "sentinel-2" | "unknown";

interface BoundingBox {
  x: number;       // pixels from left of tile
  y: number;       // pixels from top of tile
  width: number;
  height: number;
  geo_bbox?: GeoBBox;  // EPSG:4326, optional
}

interface GeoBBox {
  west: number; south: number; east: number; north: number;
}

interface Confidence {
  score: number;        // [0.0, 1.0]
  method: string;       // e.g., "cosine_similarity", "pixel_fraction"
  calibrated: boolean;
}

interface TileRef {
  tile_id: string;       // e.g., "paris_0012_0034_sentinel-2"
  location_id: string;   // e.g., "paris"
  date: string;          // ISO 8601 date: "2020-06-15"
  sensor: Sensor;
}
```

### 12.2 Endpoint Specifications

---

#### `GET /api/v1/health`
**Phase:** Prototype

**Response 200:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "embedding_model": "mock",
  "change_detector": "pixel_diff",
  "faiss_index_size": 15,
  "db_tiles": 15
}
```
**Error:** None (always 200 or 503 if startup failed).

---

#### `POST /api/v1/search`
**Phase:** Prototype

**Request:**
```json
{
  "query_type": "text",          // "text" | "image"
  "query_text": "urban expansion near river",   // required if query_type=text
  "query_image_b64": null,       // base64-encoded PNG/JPEG, required if query_type=image
  "filters": {
    "location": "paris",         // optional; partial match
    "date_from": "2018-01-01",   // optional; ISO 8601
    "date_to": "2021-12-31",     // optional
    "sensor": "sentinel-2"       // optional; or "any"
  },
  "top_k": 10                    // optional; default 10; max 50
}
```

**Validation:**
- Exactly one of `query_text` or `query_image_b64` must be non-null.
- `query_text`: 3–500 chars.
- `date_from` ≤ `date_to` if both provided.
- `top_k`: 1–50.

**Response 200:**
```json
{
  "query_id": "uuid4-string",
  "results": [
    {
      "rank": 1,
      "tile_ref": { "tile_id": "...", "location_id": "paris", "date": "2020-06-15", "sensor": "sentinel-2" },
      "confidence": { "score": 0.87, "method": "cosine_similarity", "calibrated": false },
      "thumbnail_url": "/api/v1/images/paris_0012_0034_sentinel-2/2020-06-15/thumbnail",
      "available_dates": ["2018-03-10", "2020-06-15"],
      "geo_bbox": { "west": 2.29, "south": 48.85, "east": 2.31, "north": 48.87 }
    }
  ],
  "total": 10,
  "query_ms": 142
}
```

**Errors:**
- `422 Unprocessable Entity`: validation failure with field-level messages.
- `503 Service Unavailable`: embedding model not initialized.

---

#### `GET /api/v1/images/{tile_id}/{date}`
**Phase:** Prototype

**Path params:** `tile_id` (string), `date` (ISO 8601 date string).

**Query params:** `bands` (optional, comma-separated band names, default="RGB"), `format` (optional, "png"|"jpeg", default="png").

**Response 200:** Raw image bytes (`Content-Type: image/png` or `image/jpeg`).

**Errors:**
- `404`: Tile or date not found.
- `422`: Invalid date format.

---

#### `GET /api/v1/images/{location_id}/dates`
**Phase:** Prototype

**Response 200:**
```json
{
  "location_id": "paris",
  "dates": ["2018-03-10", "2020-06-15"],
  "sensor": "sentinel-2"
}
```

---

#### `POST /api/v1/comparison`
**Phase:** Prototype

**Request:**
```json
{
  "location_id": "paris",
  "tile_id": "paris_0012_0034_sentinel-2",
  "date_before": "2018-03-10",
  "date_after": "2020-06-15",
  "sensor": "sentinel-2"
}
```

**Response 200:**
```json
{
  "comparison_id": "uuid4-string",
  "before": {
    "tile_ref": { "tile_id": "...", "date": "2018-03-10", ... },
    "image_url": "/api/v1/images/paris_0012_0034_sentinel-2/2018-03-10",
    "cloud_cover_pct": null
  },
  "after": {
    "tile_ref": { "tile_id": "...", "date": "2020-06-15", ... },
    "image_url": "/api/v1/images/paris_0012_0034_sentinel-2/2020-06-15",
    "cloud_cover_pct": null
  },
  "coregistered": true
}
```

**Errors:**
- `404`: Either date not available for tile.
- `409`: Tiles cannot be co-registered (Phase B; not raised in Phase A).

---

#### `POST /api/v1/change-detection`
**Phase:** Prototype

**Request:**
```json
{
  "comparison_id": "uuid4-string"   // from POST /comparison
}
```
OR inline:
```json
{
  "location_id": "paris",
  "tile_id": "paris_0012_0034_sentinel-2",
  "date_before": "2018-03-10",
  "date_after": "2020-06-15"
}
```

**Response 200:**
```json
{
  "job_id": "uuid4-string",
  "status": "completed",
  "mask_url": "/api/v1/masks/uuid4-string.png",
  "bounding_boxes": [
    { "x": 45, "y": 120, "width": 60, "height": 40, "label": "change", "confidence": { "score": 0.74, "method": "pixel_fraction", "calibrated": false } }
  ],
  "summary": {
    "location": "Paris, France",
    "date_before": "2018-03-10",
    "date_after": "2020-06-15",
    "change_type": "unknown",              // "unknown" until semantic CD is implemented
    "earliest_detectable_change": null,    // Phase B+
    "confidence": { "score": 0.74, "method": "pixel_fraction", "calibrated": false },
    "changed_pixel_fraction": 0.082,
    "source_provenance": "OSCD/synthetic_fixture_v0.1",
    "detector": "pixel_diff"
  },
  "processing_ms": 88
}
```

**Errors:**
- `404`: `comparison_id` not found.
- `422`: Inline params invalid.
- `500`: Inference failure (with error detail).

---

#### `GET /api/v1/masks/{mask_filename}`
**Phase:** Prototype

**Response 200:** Raw PNG bytes of the binary change mask (white = change, black = no change).

---

### 12.3 Phase B Additional Endpoints (planned, not in prototype)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/ingest` | Trigger incremental ingestion of new scene |
| `GET /api/v1/locations` | List all indexed locations with spatial filter |
| `POST /api/v1/cluster` | Cluster visually similar locations |
| `POST /api/v1/feedback` | Submit analyst feedback on a result |
| `GET /api/v1/audit/{query_id}` | Retrieve full audit trail for a query |

---

## 13. Team Workstreams and Ownership Boundaries

### WS-1: Setup, Repository, and Shared Contracts
**Skills:** DevOps, Python tooling, TypeScript config, architecture.
**Prerequisites:** None (first to execute).
**Owns:**
- `setup/`, `.env.example`, `.gitignore`, `.pre-commit-config.yaml`, `pyproject.toml`, `uv.lock`
- `backend/app/models/interfaces.py`
- `data-handling/interfaces/`
- `data-handling/fixtures/`
- `docs/api_reference.md` (initial version)
- `docs/model_registry.md`

**Deliverables:**
1. Initialized Git repo with branch protection.
2. Installable Python workspace (`uv sync` runs clean).
3. Scaffolded Vite frontend (`npm run dev` serves blank app).
4. API contract document (this plan, Section 12).
5. ABCs for `EmbeddingModel`, `ChangeDetector`, `DatasetAdapter`.
6. Synthetic fixtures (2 tile pairs + ground-truth masks).
7. pytest configured with 0 failures.
8. pre-commit hooks enforced.

**Acceptance criteria:** All other workstreams can begin without WS-1 participation.

**Tests before integration:** `uv run pytest` passes; `npm run dev` builds; `pre-commit run --all-files` passes.

---

### WS-2: Frontend — Search and Application Layout
**Skills:** React, TypeScript, Vanilla CSS.
**Prerequisites:** WS-1 complete (API contract + scaffold).
**Owns:**
- `frontend/src/routes/`
- `frontend/src/layout/`
- `frontend/src/features/search/`
- `frontend/src/api/client.ts`, `search.api.ts`, `health.api.ts`
- `frontend/src/store/`
- `frontend/src/components/`
- `frontend/src/styles/`
- `frontend/src/types/api.types.ts` (shared, but WS-2 is primary owner)

**Deliverables:**
1. Working search page with query input, filter panel, and submission.
2. Result list rendering mock API response.
3. Navigation to comparison page.
4. Global CSS design system (variables, typography, layout).

**Dependencies on other workstreams:** Blocked by WS-1 (API contract). Requires WS-4 to run the backend for integration, but can mock with MSW.

**Integration points:** `src/api/search.api.ts` must match `POST /search` contract exactly.

**Acceptance criteria:** PA-6, PA-7 from prototype milestone checklist. All component tests pass.

---

### WS-3: Frontend — Temporal Visualization and Result Summary
**Skills:** React, TypeScript, Leaflet, canvas/SVG.
**Prerequisites:** WS-1 complete. WS-2 navigation in place (can be mocked).
**Owns:**
- `frontend/src/features/comparison/`
- `frontend/src/features/visualization/`
- `frontend/src/features/results/`
- `frontend/src/features/summary/`
- `frontend/src/api/images.api.ts`, `comparison.api.ts`, `changeDetection.api.ts`

**Deliverables:**
1. Split/swipe before/after viewer.
2. Change mask overlay (semi-transparent).
3. Bounding box SVG layer.
4. Leaflet map centred on tile bbox.
5. Result summary panel with all fields.

**Dependencies:** WS-1 (types), WS-4 (backend for integration), WS-2 (routing shell).

**Acceptance criteria:** PA-8, PA-9, PA-10. Visual regression test screenshots committed for mask and bbox rendering.

---

### WS-4: Backend — API and Orchestration
**Skills:** Python, FastAPI, Pydantic, asyncio.
**Prerequisites:** WS-1 complete (interfaces + fixture data).
**Owns:**
- `backend/main.py`, `config.py`, `logging_config.py`
- `backend/app/api/`
- `backend/app/schemas/`
- `backend/app/services/`
- `backend/app/vector_store/`
- `backend/app/db/`
- `backend/app/geospatial/`
- `backend/app/postprocessing/`
- `backend/app/exceptions.py`

**Deliverables:**
1. All prototype endpoints returning correct responses (using fixture data).
2. FAISS flat index pre-seeded with mock embeddings for 5 fixture tiles.
3. SQLite DB pre-seeded with fixture metadata.
4. `pixel_diff.py` baseline change detector integrated.
5. `mock_embedding.py` integrated.
6. All backend unit + integration tests passing.

**Dependencies:** WS-1 (interfaces, fixtures). Does not depend on WS-2 or WS-3.

**Acceptance criteria:** PA-1 through PA-5. `pytest backend/tests/ -v` → all pass.

---

### WS-5: Data-Handling — Abstractions and OSCD Integration
**Skills:** Python, rasterio, GDAL, geospatial processing, numpy.
**Prerequisites:** WS-1 complete (interfaces). OSCD download is a prerequisite for Phase B portion.
**Owns:**
- `data-handling/interfaces/` (implementation, not ABCs — those are WS-1)
- `data-handling/adapters/`
- `data-handling/preprocessing/`
- `data-handling/geospatial/`
- `data-handling/ingestion/`
- `data-handling/fixtures/` (extend, not create — WS-1 creates initial)
- `data-handling/splits/`
- `data-handling/tests/`

**Phase A deliverable:** `FixtureTileAdapter` implementing `DatasetAdapter` using synthetic fixtures. All tests passing.

**Phase B deliverables (after OSCD download + verification):**
1. `oscd_loader.py` reading band TIFFs correctly.
2. `oscd_metadata.py` normalizing metadata to `TileMetadata`.
3. `pipeline.py` preprocessing pipeline for Sentinel-2.
4. `ingest_oscd.py` CLI producing a populated FAISS index and SQLite DB.

**Integration points:** Backend imports `DatasetAdapter` from `data-handling/interfaces/`. Ingestion script is run offline before backend starts.

**Acceptance criteria (Phase A):** `pytest data-handling/tests/` passes using fixture data. `FixtureTileAdapter.load_tile()` returns correct numpy arrays.

**Acceptance criteria (Phase B):** Ingestion script completes on OSCD without errors. Backend can retrieve OSCD tiles by `tile_id`.

---

### WS-6: Model Evaluation — Semantic Retrieval and Change Detection
**Skills:** Python, PyTorch, ML evaluation, OSCD domain knowledge.
**Prerequisites:** WS-1 (fixtures), WS-5 Phase B (OSCD data), Checkpoint CP-1 and CP-2 approval.
**Owns:**
- `backend/app/models/embedding/remote_clip.py` (after CP-1 approval)
- `backend/app/models/change_detection/bit_cd.py` (after CP-2 approval)
- `backend/app/models/change_detection/mamba_cd.py` (Phase C, after CP-3 approval)
- `scripts/evaluate.py`
- `docs/adr/`

**Phase B deliverables:**
1. RemoteCLIP (or approved model) adapter implementing `EmbeddingModel`.
2. BIT-CD or ChangeFormer adapter implementing `ChangeDetector`.
3. Evaluation script with Recall@K for retrieval, F1/IoU for change detection.
4. Evaluation results on OSCD test split.
5. ADR document for each model selection decision.

**Dependencies:** CP-1 and CP-2 approvals from user. WS-5 OSCD ingestion complete.

**Acceptance criteria:** Retrieval Recall@5 ≥ 0.5 on held-out OSCD pairs. Change detection F1 ≥ 0.4 on OSCD test cities (baselines; targets to be refined after CP-2 discussion).

---

### Ownership Matrix Summary

| Module | WS-1 | WS-2 | WS-3 | WS-4 | WS-5 | WS-6 |
|--------|------|------|------|------|------|------|
| `setup/` | ✅ | | | | | |
| `frontend/layout/`, `routes/`, `store/` | | ✅ | | | | |
| `frontend/features/search/` | | ✅ | | | | |
| `frontend/features/comparison/`, `visualization/`, `summary/` | | | ✅ | | | |
| `frontend/api/` | | ✅(client,search) | ✅(images,compare,cd) | | | |
| `backend/app/api/`, `schemas/`, `services/` | | | | ✅ | | |
| `backend/app/models/interfaces.py` | ✅ | | | | | |
| `backend/app/models/embedding/mock` | | | | ✅ | | |
| `backend/app/models/embedding/remote_clip` | | | | | | ✅ |
| `backend/app/models/change_detection/pixel_diff` | | | | ✅ | | |
| `backend/app/models/change_detection/bit_cd, mamba_cd` | | | | | | ✅ |
| `backend/app/vector_store/`, `db/`, `postprocessing/` | | | | ✅ | | |
| `data-handling/interfaces/` (ABCs) | ✅ | | | | | |
| `data-handling/adapters/`, `preprocessing/`, `ingestion/` | | | | | ✅ | |
| `data-handling/fixtures/` (initial) | ✅ | | | | ✅(extend) | |
| `scripts/evaluate.py`, `docs/adr/` | | | | | | ✅ |

---

## 14. Dependency Graph and Parallel Execution Opportunities

### 14.1 Linear Dependencies (must be sequential)

```
WS-1 Setup (S1–S10)
  ↓
┌──────────────────────────────────────────────────────────┐
│  Parallel after WS-1:                                   │
│  WS-2 (Frontend Search)                                 │
│  WS-3 (Frontend Visualization) [waits for WS-2 shell]  │
│  WS-4 (Backend API)                                     │
│  WS-5 Phase A (Fixture Adapter)                         │
└──────────────────────────────────────────────────────────┘
  ↓ (all Phase A workstreams complete)
Phase A Integration Test (PA-1 to PA-12)
  ↓ (user approval)
┌──────────────────────────────────────────────────────────┐
│  Parallel in Phase B:                                   │
│  WS-5 Phase B (OSCD ingestion) — OSCD must be downloaded│
│  WS-6 Phase B (Model eval) — needs WS-5 OSCD data      │
│  WS-2/WS-3 Phase B (frontend enhancements)             │
│  WS-4 Phase B (backend enhancements)                   │
└──────────────────────────────────────────────────────────┘
  ↓ CP-1 approval → WS-6 integrates retrieval model
  ↓ CP-2 approval → WS-6 integrates change detection model
  ↓ Phase B integration test
  ↓ (user approval)
Phase C: Mamba architecture, CP-3, SAR support
```

### 14.2 Critical Path

```
WS-1 Setup → WS-4 Backend (prototype endpoints + pixel diff) → Phase A Integration → User Approval → Phase B
```

WS-4 is the critical path bottleneck for the prototype because the frontend can mock API calls, but the end-to-end prototype test requires real backend responses.

### 14.3 Parallel Opportunities

| Work pair | Can run in parallel? | Condition |
|-----------|---------------------|-----------|
| WS-2 + WS-4 | ✅ Yes | Both use agreed API contract; WS-2 mocks backend with MSW |
| WS-3 + WS-4 | ✅ Yes | Same as above |
| WS-2 + WS-3 | ⚠️ Partial | WS-3 needs WS-2's routing shell; mock it with a stub route |
| WS-5 + WS-4 | ✅ Yes | WS-4 uses FixtureTileAdapter; WS-5 implements it independently |
| WS-5 OSCD + WS-6 eval | ❌ No | WS-6 needs ingested OSCD data from WS-5 |
| WS-6 CP-1 + WS-6 CP-2 | ❌ No | CP-1 approval must precede embedding integration which affects CP-2 evaluation |

---

## 15. Detailed, Ordered Implementation Roadmap

### Stage 0: Setup (WS-1 only, ~1–2 days)

| Step | Tasks | Deliverables | Verification |
|------|-------|--------------|-------------|
| 0.1 | Git init, `.gitignore`, branch protection, `README.md` | Repo accessible | `git log` shows initial commit |
| 0.2 | `pyproject.toml` (uv workspace), `backend/pyproject.toml`, `data-handling/pyproject.toml` | `uv sync` succeeds | No import errors |
| 0.3 | Vite scaffold: `cd frontend && npx create-vite@latest ./ --template react-ts` | `npm run dev` shows blank Vite page | Browser renders Vite default |
| 0.4 | `.env.example`, `config.py` (Pydantic-Settings, reads from `.env`) | Config loads without errors | Unit test: config parses valid `.env.example` |
| 0.5 | `backend/app/models/interfaces.py`: `EmbeddingModel`, `ChangeDetector` ABCs | Importable | `python -c "from app.models.interfaces import EmbeddingModel"` |
| 0.6 | `data-handling/interfaces/`: `DatasetAdapter`, `TileProvider` ABCs | Importable | Same pattern |
| 0.7 | `data-handling/fixtures/synthetic_tiles/`: 2 PNG pairs + masks; `mock_metadata.json` | Files exist | pytest fixture loads them |
| 0.8 | `pre-commit` config, hooks pass on all current files | No hook failures | `pre-commit run --all-files` |
| 0.9 | `docs/api_reference.md` (this document's Section 12) | Published | Review by all WS leads |
| 0.10 | pytest config for backend + data-handling | `pytest` runs 0 tests, 0 failures | CI-ready |

**Approval required:** Yes — user must confirm setup is satisfactory before parallel work begins.

---

### Stage 1: Phase A Parallel Development (~3–5 days)

#### Stage 1a: Backend (WS-4, starts after Stage 0)

| Step | Tasks | Files | Verification |
|------|-------|-------|-------------|
| 1a.1 | `logging_config.py`, `exceptions.py`, `main.py` skeleton with health endpoint | `main.py`, `app/api/v1/health.py` | PA-1: curl health |
| 1a.2 | Pydantic schemas for all endpoints | `app/schemas/` | Import without errors |
| 1a.3 | `mock_embedding.py`, `pixel_diff.py` (implements interfaces) | `app/models/` | Unit tests for both |
| 1a.4 | SQLite setup, ORM models, seed fixture tiles | `app/db/` | DB has 5 rows in `tiles` |
| 1a.5 | FAISS flat index, seed with mock embeddings for fixtures | `app/vector_store/` | `faiss_store.search()` returns fixture tile IDs |
| 1a.6 | `retrieval_service.py`, `image_service.py`, `comparison_service.py` | `app/services/` | Unit tests with mocked dependencies |
| 1a.7 | `change_detection_service.py`, `postprocessing/` | `app/services/`, `app/postprocessing/` | Pixel diff returns valid mask PNG |
| 1a.8 | `summary_service.py`, assemble `SummaryResponse` | `app/services/` | Unit test with fixture data |
| 1a.9 | All API endpoints wired to services | `app/api/v1/` | PA-2 through PA-5 |
| 1a.10 | Backend integration tests | `backend/tests/integration/` | All PA assertions pass |

#### Stage 1b: Frontend Search (WS-2, starts after Stage 0)

| Step | Tasks | Files | Verification |
|------|-------|-------|-------------|
| 1b.1 | CSS design system: variables, typography, global reset | `styles/` | Visual review |
| 1b.2 | `AppShell`, `TopBar`, `Sidebar` | `layout/` | Renders on all routes |
| 1b.3 | React Router setup, all page routes | `routes/`, `App.tsx` | Navigation works |
| 1b.4 | `SearchBar`, `QueryTypeToggle`, `FilterPanel` | `features/search/` | Form submits valid data |
| 1b.5 | `SearchStore` Zustand slice | `store/` | State transitions correct |
| 1b.6 | `search.api.ts` with MSW mock returning fixture response | `api/` | PA-6 |
| 1b.7 | `ResultList`, `ResultCard`, `ConfidenceBadge` | `features/results/` | PA-7 |
| 1b.8 | Component tests for all above | `tests/components/` | Vitest passes |

#### Stage 1c: Frontend Visualization (WS-3, starts after 1b.3 routing shell exists)

| Step | Tasks | Files | Verification |
|------|-------|-------|-------------|
| 1c.1 | `ComparisonViewer`, `DateSelector`, `SliderSplit` | `features/comparison/` | PA-8 |
| 1c.2 | `ImageTileRenderer`, `MapView` (Leaflet) | `features/visualization/` | Map centres on bbox |
| 1c.3 | `ChangeMaskOverlay` (canvas blend) | `features/visualization/` | PA-9 |
| 1c.4 | `BoundingBoxLayer` (SVG) | `features/visualization/` | Boxes rendered correctly |
| 1c.5 | `ResultSummaryPanel`, `ChangeTimeline` | `features/summary/` | PA-10 |
| 1c.6 | Component tests; visual snapshot tests for overlay | `tests/components/` | Vitest passes |

#### Stage 1d: Data-Handling Fixture Adapter (WS-5, starts after Stage 0)

| Step | Tasks | Files | Verification |
|------|-------|-------|-------------|
| 1d.1 | `FixtureTileAdapter` implementing `DatasetAdapter` | `adapters/fixtures/` | Unit test: loads 2 tile pairs |
| 1d.2 | Tests for fixture adapter | `data-handling/tests/` | pytest passes |
| 1d.3 | Register `FixtureTileAdapter` in backend `main.py` dependency injection | `backend/main.py` | Backend uses fixture data |

---

### Stage 2: Phase A Integration (~1 day)

| Step | Tasks | Verification |
|------|-------|-------------|
| 2.1 | Start backend (`uvicorn`), start frontend (`npm run dev`), disable MSW mocks | Both servers running |
| 2.2 | Manual walkthrough: submit query → view results → open comparison → view mask + summary | PA-1 through PA-12 all pass |
| 2.3 | Run all backend tests against live server | `pytest` passes |
| 2.4 | Run all frontend tests | `npx vitest run` passes |
| 2.5 | Document any outstanding issues | Issues list in `docs/` |

**Approval required:** Yes — user reviews working prototype demo before Phase B begins.

---

### Stage 3: Phase B — OSCD Integration and Real Models (~5–10 days, requires OSCD download)

| Step | Workstream | Tasks |
|------|-----------|-------|
| 3.1 | WS-5 | Verify OSCD directory; document in `oscd_schema.md` |
| 3.2 | WS-5 | Implement `oscd_loader.py`, `oscd_metadata.py`; tests on 2 cities |
| 3.3 | WS-5 | Implement `preprocessing/pipeline.py` for Sentinel-2 |
| 3.4 | WS-5 | Implement `ingest_oscd.py` CLI; run on 2 OSCD city pairs |
| 3.5 | WS-4 | Replace `FixtureTileAdapter` with `OSCDAdapter` in production config |
| 3.6 | WS-6 | **CP-1: Present retrieval model comparison. Await approval.** |
| 3.7 | WS-6 | Implement approved embedding model adapter |
| 3.8 | WS-5 | Run full OSCD ingestion with real embeddings; rebuild FAISS index |
| 3.9 | WS-6 | **CP-2: Present change detection model comparison. Await approval.** |
| 3.10 | WS-6 | Implement approved change detector adapter |
| 3.11 | WS-4 | Wire approved models into backend via config switch |
| 3.12 | All | Phase B integration test; evaluation script on OSCD test split |

---

### Stage 4: Phase C — Mamba Architecture (~open-ended, requires GPU and CP-3 approval)

Steps deferred to post-Phase B planning session. Requires CP-3 approval on architecture, training strategy, and compute budget.

---

## 16. Testing Strategy and Acceptance Criteria

### 16.1 Unit Tests

| Scope | Tool | Criteria |
|-------|------|----------|
| `EmbeddingModel` implementations | pytest | Correct output shape, dtype float32, L2-normalized |
| `ChangeDetector` implementations | pytest | Mask shape = input shape; mask dtype bool; confidence ∈ [0,1] |
| `DatasetAdapter` implementations | pytest | Returns correct numpy shape for known fixture tile |
| `retrieval_service` | pytest + mock | Returns `top_k` results; ranking is descending by score |
| `change_detection_service` | pytest + mock | Returns `ChangeDetectionResult` with non-empty mask |
| `postprocessing/bbox_extraction` | pytest | Bboxes non-overlapping; area > min_area |
| `postprocessing/mask_refinement` | pytest | Output mask has same shape as input; small noise removed |
| Frontend Zustand stores | Vitest | State transitions correct for all defined actions |
| Frontend validation | Vitest | Invalid inputs rejected; valid inputs pass |

### 16.2 API Contract Tests

For each endpoint: send a request exactly matching the contract schema; assert response matches the response schema field-by-field. Run against the live FastAPI app using `httpx.AsyncClient`. These tests act as the binding specification between frontend and backend.

### 16.3 Frontend Component Tests

Using `@testing-library/react` + Vitest:
- Render each feature component with mock props and MSW-intercepted API.
- Assert that key DOM elements are present.
- Assert that user interactions (button clicks, form submissions) trigger the correct API calls.
- Snapshot tests for `ChangeMaskOverlay` and `BoundingBoxLayer` to catch visual regressions.

### 16.4 Data-Loading Tests

- `test_oscd_loader.py`: For each fixture tile, assert output shape, dtype, and value range [0, 1].
- `test_preprocessing.py`: Apply each preprocessing step to fixture tile; assert output is valid.
- Dataset integrity check: assert all expected tile IDs are present after ingestion.
- Image-pair validation: assert before/after tiles have identical shape and CRS.

### 16.5 Model Inference Tests

- `test_embedding_model.py`: Encode 2 different texts; assert cosine similarity < 1.0. Encode same text twice; assert identical output.
- `test_change_detector.py`: Pass identical before/after arrays; assert changed_pixel_fraction ≈ 0. Pass completely different arrays; assert changed_pixel_fraction > 0.5.

### 16.6 End-to-End Prototype Tests

Using `httpx` (or Playwright for browser-level):
1. POST `/search` → assert 200, result count = expected.
2. GET `/images/{tile_id}/{date}` → assert 200, Content-Type image/png.
3. POST `/comparison` → assert 200, both image URLs resolve.
4. POST `/change-detection` → assert 200, mask_url resolves, bboxes non-empty.

### 16.7 Evaluation Metrics (Phase B, not prototype)

**Semantic Retrieval:**
- Recall@K (K=1, 5, 10): fraction of queries where the ground-truth tile appears in the top-K results.
- Mean Reciprocal Rank (MRR).
- Evaluated on OSCD test-split cities only (not seen during ingestion).

**Change Detection:**
- Pixel-level F1-score, Precision, Recall (binary: change / no-change).
- IoU (Intersection over Union) of predicted vs. ground-truth change mask.
- Per-city breakdown to identify geographic generalization.
- Reported separately for: baseline (pixel diff), Phase B model, Phase C model.

**Important:** Confidence scores from the prototype are heuristics, not calibrated probabilities. Do not report them as accuracy metrics. Calibration must be performed against OSCD ground truth before confidence scores are presented as evidence of model accuracy.

### 16.8 Regression Tests

When any of the following change, re-run the full test suite plus evaluation:
- Preprocessing pipeline steps.
- Embedding model weights or version.
- Change detector model weights or version.
- FAISS index rebuild.
- Tile size or tiling parameters.

---

## 17. Risks, Limitations, and Mitigation Strategies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OSCD directory structure differs from assumptions | Medium | High | `verify_dataset.py` script run immediately after download; `oscd_schema.md` updated before any adapter code is written |
| Mamba SSM requires CUDA; no GPU available | Medium | High (Phase C) | All Phase A/B models run on CPU. Mamba deferred until hardware confirmed at CP-3 |
| RemoteCLIP weights unavailable or licence incompatible | Low | Medium | CLIP (OpenAI) is fallback; always maintain mock adapter |
| False positive rate from pixel-diff baseline is high | High | Medium (Phase A) | Documented as known limitation; not presented as final quality; Phase B model addresses this |
| Co-registration drift between OSCD dates | Medium | Medium | ECC algorithm in `coregistration.py`; validated with image-pair test; report drift metric |
| Frontend image rendering performance (large tiles) | Low | Low | Tile size capped at 512×512 in Phase A; progressive loading in Phase B |
| API schema mismatch between frontend and backend | Medium | High | API contract tests are mandatory and run on every PR |
| Seasonal false alarms incorrectly flagged as real changes | High | High | False-alarm suppression is a Phase B/C feature; documented as limitation in prototype |
| Model weights too large for practical local storage | Low | Medium | `model_registry.md` documents size; minimum hardware requirements documented |
| SQLite SpatiaLite not available on target OS | Low | Low | Spatial queries deferred to Phase B; Phase A uses pure SQLite; document SpatiaLite installation |
| Cross-sensor change detection produces nonsensical results | High | High | Sensor type validation in `change_detection_service.py`; cross-sensor CD blocked until Phase C |

---

## 18. Final Checklist for Determining When Each Phase Is Complete

### Phase A Complete When:
- [ ] All 12 prototype milestones (PA-1 to PA-12) verified.
- [ ] `pytest backend/tests/` exits 0 with ≥ 80% service coverage.
- [ ] `npx vitest run` exits 0 with ≥ 70% feature coverage.
- [ ] End-to-end walkthrough completed successfully by a user not involved in development.
- [ ] All open Q1–Q10 questions from Section 2.2 that affect Phase A answered and documented.
- [ ] `docs/data_provenance.md` documents licence and origin of all sample images used.
- [ ] `docs/model_registry.md` records all model choices (including mock) with rationale.
- [ ] User approval received.

### Phase B Complete When:
- [ ] OSCD ingested without errors; `verify_dataset.py` passes.
- [ ] OSCD test-split cities are excluded from the retrieval index.
- [ ] CP-1 approval obtained; approved embedding model integrated and tested.
- [ ] CP-2 approval obtained; approved change detector integrated and tested.
- [ ] Retrieval Recall@5 ≥ 0.5 on OSCD test split (or documented if not achievable, with explanation).
- [ ] Change detection F1 ≥ 0.4 on OSCD test cities (or documented).
- [ ] Confidence scores not presented as calibrated probabilities without calibration evidence.
- [ ] Preprocessing pipeline handles all Sentinel-2 OSCD tiles without errors.
- [ ] Incremental ingestion tested: adding a new scene does not corrupt the existing index.
- [ ] User approval received.

### Phase C Complete When:
- [ ] CP-3 approval obtained; Mamba architecture finalized and justified.
- [ ] Mamba models trained and validated on OSCD.
- [ ] Temporal state persistence tested across multiple ingestion runs.
- [ ] Semantic change detection classifies at least 3 change types correctly.
- [ ] False-alarm suppression reduces false positive rate vs. Phase B baseline (documented metric).
- [ ] End-to-end offline operation verified (no network calls after model weights are on disk).
- [ ] User approval received.

---

*Plan version: 1.0 | Date: 2026-09-30 | Status: Approved.*
