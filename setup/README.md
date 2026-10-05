# TerraEyes Setup & Environment Guide

This folder contains project configuration, environment guidelines, model registries, and developer standards.

## Setup Instructions

### 1. Python Environment (3.11+)
Using `uv` (recommended):
```bash
uv venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
uv pip install -r setup/requirements-dev.txt
```

Alternatively, using `pip`:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r setup/requirements-dev.txt
```

### 2. Frontend Setup (Node 18+)
```bash
cd frontend
npm install
npm run dev
```

### 3. CLIP ViT-B/32 Weights (Phase B)

The approved retrieval adapter reads a local checkpoint and does not download
weights while the API is running. Install the Python dependencies, then fetch
the OpenAI ViT-B/32 checkpoint once during setup:

```powershell
python -m pip install -r setup/requirements.txt
python scripts/download_models.py --model clip_vit_b32 --output-dir ./models
python data-handling/ingestion/ingest_oscd.py --oscd-dir ./images --output-dir ./data
python scripts/build_index.py --manifest ./data/oscd_tiles/manifest.json --oscd-dir ./images --checkpoint-path ./models/clip_vit_b32.pt
```

Set `TERRAEYES_EMBEDDING_MODEL=clip_vit_b32` and
`TERRAEYES_CLIP_CHECKPOINT_PATH=./models/clip_vit_b32.pt` in `.env`. Keep the
model file local; it is excluded from source control.
The index builder reads the train/test split and indexes only the 14 training
cities. A separate held-out location-name sanity check can be run with
`python scripts/evaluate_retrieval.py`; it uses a temporary test-only candidate
set and does not add test cities to the production index.

### 4. Verification
Run test suites:
```bash
pytest
```
Measure backend service coverage against the Phase A target:
```bash
pytest backend/tests --cov=backend/app/services --cov-report=term-missing --cov-fail-under=80
```
Run linting:
```bash
ruff check .
mypy backend/app data-handling/interfaces
```

### Phase A Prototype Walkthrough

Phase A uses the deterministic mock embedding catalog and the real pixel-difference
detector. Start the backend from the repository root in PowerShell:

```powershell
$env:TERRAEYES_EMBEDDING_MODEL = "mock"
python -m uvicorn main:app --app-dir backend --reload
```

In a second terminal, start the UI:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:3000`, search, inspect a result, and switch to the image
comparison view to see the generated change mask and analyst summary. The mock
embedding scores are demonstration rankings, not semantic retrieval accuracy.
The Norcia catalog entry is a synthetic demonstration tile when no matching OSCD
scene is available. Other pairs can also produce no change boxes with the Otsu
baseline; the detector's actual output is reported by the API.

### Train and benchmark the temporal SSM change detector

The repository includes a compact PyTorch implementation of Mamba's selective
state-space recurrence. It runs on CPU and accepts ordered, co-registered image
sequences. Train only on the official OSCD train cities, reserving three of
those cities for validation and threshold selection:

```powershell
python scripts/train_mamba_cd.py --epochs 12 --batch-size 2
```

The trained checkpoint is written to
`models/change_detection_candidates/mamba_oscd_best.pt`; the training history
is written to `data/evaluation/mamba_oscd_training.json`. Then compare it with
the Otsu baseline on the official OSCD test-city split, which is not used for
fitting or threshold selection:

```powershell
python scripts/benchmark_mamba_cd.py
```

The JSON report includes per-city and macro/micro F1/IoU, CPU/GPU scene
latency, tile throughput, model size, and comparison with Otsu. The model can
be enabled in the API after training by setting
`TERRAEYES_CHANGE_DETECTOR=mamba_cd` and
`TERRAEYES_MAMBA_CHECKPOINT_PATH=./models/change_detection_candidates/mamba_oscd_best.pt`.
It stores per-tile, per-date recurrent and convolution context in the configured
HDF5 temporal store. Keep `pixel_diff` as the default unless the held-out
benchmark supports changing it.

The current OSCD test report is exploratory: a temporal-only prototype was
benchmarked on the test cities before the spatial-scan version was finalized.
For a clean model-selection result, reserve a new geographic holdout that has
not informed architecture iteration.

OSCD provides two acquisitions and binary stable/change masks, not semantic
land-cover transition labels. The benchmark therefore measures binary
multi-temporal change segmentation; it cannot validate labels such as
vegetation-to-building. Mamba's scan is a plain PyTorch reference recurrence,
not the fused CUDA kernel from the official `mamba-ssm` package.

### OSCD Dataset Prototype

The default backend configuration uses the local CLIP checkpoint and the
training-only OSCD image index. With `models/clip_vit_b32.pt`,
`data/faiss.index.npz`, `data/faiss.index.meta.json`, and the ingested
`data/terraeyes.db` present, start the backend without the Phase A mock override:

```powershell
python -m uvicorn main:app --app-dir backend --reload
```

Keep `VITE_API_BASE_URL` set to the API origin (`http://localhost:8000`), then
start the frontend as above. The health view should report `clip_vit_b32`, 100
indexed training tiles, and 200 dated tile records. The test split is excluded
from that retrieval index.
