import pytest
import numpy as np
from pathlib import Path
# test/test_io_utils.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples")))

from io_utils import load_sample_image, SAMPLE_DATA_DIR


@pytest.mark.parametrize("name", ["camera", "moon", "coins"])
def test_load_sample_image(name):
    shape = (128, 128)
    img = load_sample_image(name=name, shape=shape)

    # Type and shape check
    assert isinstance(img, np.ndarray)
    assert img.shape == shape
    assert img.dtype == np.float32

    # Value range check
    assert 0.0 <= img.min() <= 1.0
    assert 0.0 <= img.max() <= 1.0

    # Cached file existence check (absolute path!)
    path = SAMPLE_DATA_DIR / f"{name}_{shape[0]}x{shape[1]}.png"
    assert path.exists()
