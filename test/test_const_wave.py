import pytest
import numpy as np
from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.graph.prior.const_wave import ConstWave
from gpie.core.backend import move_array_to_current_backend


def test_const_wave_to_backend_roundtrip():
    cp = pytest.importorskip("cupy")  # Skip test if CuPy is not available

    # Setup: define shape, dtype, and data
    batch_size = 2
    event_shape = (3, 3)
    shape = (batch_size,) + event_shape
    dtype = np.complex64

    # Initialize in NumPy backend
    backend.set_backend(np)
    data_np = np.ones(shape, dtype=dtype)
    cw = ConstWave(data=data_np, batch_size=batch_size, event_shape=event_shape)

    assert cw._data.shape == shape
    assert cw._data.dtype == dtype
    assert cw.dtype == dtype

    # Switch to CuPy and convert data
    backend.set_backend(cp)
    cw.to_backend()
    assert isinstance(cw._data, cp.ndarray)
    assert cw._data.dtype == cp.complex64
    assert cw.dtype == cp.complex64

    # Switch back to NumPy and reconvert
    backend.set_backend(np)
    cw.to_backend()
    assert isinstance(cw._data, np.ndarray)
    assert cw._data.dtype == np.complex64
    assert cw.dtype == np.complex64

    # Check that the content remains consistent (should be all 1.0)
    assert np.allclose(cw._data, 1.0)
