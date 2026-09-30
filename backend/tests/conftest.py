"""Pytest fixtures for TerraEyes backend."""
import pytest
import numpy as np
from pathlib import Path
import sys

# Ensure backend and data-handling are importable
backend_root = Path(__file__).resolve().parents[1]
data_handling_root = backend_root.parent / "data-handling"
for p in (backend_root, data_handling_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.models.embedding.mock_embedding import MockEmbeddingModel
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector
from app.vector_store.faiss_store import FaissVectorStore
from app.db.connection import DatabaseManager
from app.db.repositories.tile_repo import TileRepository
from app.services.image_service import ImageService
from app.services.comparison_service import ComparisonService
from app.services.summary_service import SummaryService
from app.services.change_detection_service import ChangeDetectionService
from app.services.retrieval_service import RetrievalService
from app.services.query_orchestrator import QueryOrchestrator


@pytest.fixture
def mock_embedding():
    return MockEmbeddingModel(dim=128)


@pytest.fixture
def pixel_diff_detector():
    return PixelDiffChangeDetector()


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_terraeyes.db"
    mgr = DatabaseManager(str(db_file))
    mgr.init_db()
    return mgr


@pytest.fixture
def tile_repo(temp_db):
    return TileRepository(temp_db)


@pytest.fixture
def vector_store():
    return FaissVectorStore(dim=128)


@pytest.fixture
def sample_tile_pair():
    np.random.seed(42)
    before = np.zeros((256, 256, 3), dtype=np.float32)
    before[:, :] = [0.2, 0.5, 0.2]

    after = before.copy()
    # Add a changed building block
    after[100:150, 100:150] = [0.8, 0.2, 0.1]
    return before, after
