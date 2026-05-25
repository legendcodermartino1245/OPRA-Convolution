import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def fft_size():
    return 1024


@pytest.fixture(scope="session")
def magnitude(fft_size):
    return np.ones(fft_size // 2 + 1, dtype=np.float64)


@pytest.fixture(scope="session")
def headroom_db():
    return 6.0

