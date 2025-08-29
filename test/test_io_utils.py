import pytest
import numpy as np
from pathlib import Path
from gpie.examples.io_utils import load_sample_image, SAMPLE_DATA_DIR  # <- 追加

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
    path = SAMPLE_DATA_DIR / f"{name}.png"
    assert path.exists()
