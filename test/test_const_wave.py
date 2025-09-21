import pytest
import numpy as np
from gpie.core import backend
from gpie.graph.prior.const_wave import ConstWave
from gpie.core.types import PrecisionMode
from gpie.core.uncertain_array import UncertainArray
from gpie.core.rng_utils import get_rng


@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_const_wave_shape_inference_and_broadcast(dtype):
    backend.set_backend(np)
    # shape matches (broadcast from event_shape only)
    cw1 = ConstWave(data=np.ones((3, 3), dtype=dtype), batch_size=2, event_shape=(3, 3))
    assert cw1._data.shape == (2, 3, 3)

    # shape already matches
    cw2 = ConstWave(data=np.ones((2, 3, 3), dtype=dtype), batch_size=2, event_shape=(3, 3))
    assert cw2._data.shape == (2, 3, 3)

    # shape mismatch
    with pytest.raises(ValueError):
        ConstWave(data=np.ones((4, 3), dtype=dtype), batch_size=2, event_shape=(3, 3))


def test_const_wave_dtype_override():
    backend.set_backend(np)
    arr = np.ones((2, 2))
    cw = ConstWave(data=arr, dtype=np.float32)
    assert cw.dtype == np.float32
    assert cw._data.dtype == np.float32


@pytest.mark.parametrize("mode", ["scalar", "array"])
def test_compute_message_modes(mode):
    backend.set_backend(np)
    data = np.ones((1, 2), dtype=np.float32)
    cw = ConstWave(data=data, precision_mode=mode)
    cw.output._precision_mode_enum = PrecisionMode(mode)

    ua = cw._compute_message(None)
    assert isinstance(ua, UncertainArray)
    if mode == "scalar":
        assert ua._scalar_precision
    else:
        assert not ua._scalar_precision
        assert np.allclose(ua.precision(), cw.large_value)


def test_compute_message_without_mode_raises():
    backend.set_backend(np)
    data = np.ones((1, 2), dtype=np.float32)
    cw = ConstWave(data=data)
    cw.output._precision_mode_enum = None
    with pytest.raises(RuntimeError):
        cw._compute_message(None)


def test_get_sample_for_output_includes_noise():
    backend.set_backend(np)
    rng = get_rng(0)
    data = np.ones((1, 3), dtype=np.float32)
    cw = ConstWave(data=data, large_value=1e6)
    sample = cw.get_sample_for_output(rng=rng)

    # Expect sample to differ slightly due to noise
    assert sample.shape[1:] == data.shape
    assert not np.allclose(sample, data)  # confirm noise added


def test_const_wave_repr():
    cw = ConstWave(data=np.ones((1, 2)), precision_mode="scalar")
    rep = repr(cw)
    assert "ConstWave" in rep
    assert "scalar" in rep or "SCALAR" in rep
