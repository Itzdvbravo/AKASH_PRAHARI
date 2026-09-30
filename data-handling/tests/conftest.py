"""Pytest configuration and shared fixtures for data-handling tests."""
import pytest
import numpy as np
from pathlib import Path
import sys

data_handling_root = Path(__file__).resolve().parents[1]
if str(data_handling_root) not in sys.path:
    sys.path.insert(0, str(data_handling_root))


@pytest.fixture
def sample_scene_array():
    """Generates a test in-memory 512x512x3 float32 array in [0, 1]."""
    np.random.seed(101)
    return np.random.uniform(0.1, 0.9, size=(512, 512, 3)).astype(np.float32)
