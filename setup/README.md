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

### 3. Verification
Run test suites:
```bash
pytest
```
Run linting:
```bash
ruff check .
mypy backend/app data-handling/interfaces
```
