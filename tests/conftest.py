from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """A deterministic, synthetic 300x300 RGB image (no real photo needed)."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)


@pytest.fixture
def sample_face_roi() -> np.ndarray:
    """A deterministic, synthetic 96x96 RGB face-crop-shaped image."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
