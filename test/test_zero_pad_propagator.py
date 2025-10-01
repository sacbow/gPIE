import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.graph.propagator.zero_pad_propagator import ZeroPadPropagator

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp
    backend_libs = [np, cp]
else:
    cp = None
    backend_libs = [np]


@pytest.mark.parametrize("xp", backend_libs)
def test_zero_pad_matmul_shapes(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)

    pad_width = ((2, 2), (1, 3))  # pad rows and cols differently
    prop = ZeroPadPropagator(pad_width)
    out = prop @ wave

    assert out.batch_size == 1
    assert out.event_shape == (4 + 2 + 2, 4 + 1 + 3)  # (8, 8)
    assert out.dtype == wave.dtype
    assert out.parent is prop


@pytest.mark.parametrize("xp", backend_libs)
def test_zero_pad_invalid_padwidth_rank(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)

    pad_width = ((1, 1),)  # only 1D, but wave is 2D
    prop = ZeroPadPropagator(pad_width)

    with pytest.raises(ValueError, match="pad_width length"):
        _ = prop @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_zero_pad_forward_and_backward(xp):
    backend.set_backend(xp)
    event_shape = (2, 2)
    wave = Wave(event_shape=event_shape, batch_size=1, dtype=xp.complex64)

    pad_width = ((1, 1), (2, 0))
    prop = ZeroPadPropagator(pad_width)
    out_wave = prop @ wave

    ua = UA.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 5.0

    # Forward: apply zero_pad
    out = prop._compute_forward({"input": ua})
    expected_shape = (1,) + tuple(dim + l + r for dim, (l, r) in zip(event_shape, pad_width))
    assert out.data.shape == expected_shape

    # Padded region should be zeros
    assert xp.all(out.data[:, 0, :] == 0)
    assert xp.all(out.data[:, -1, :] == 0)
    assert xp.all(out.data[:, :, 1] == 0)

    # Precision in padded region should be ~1e8
    large_prec = out.precision(raw=False)
    assert xp.all(large_prec[:, 0, :] > 1e6)

    # Backward: crop back to original shape
    recon = prop._compute_backward(out, exclude="input")
    assert recon.data.shape == (1, 2, 2)
    assert xp.allclose(recon.data, 5.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_zero_pad_forward_1d(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(3,), batch_size=1, dtype=xp.complex64)

    pad_width = ((2, 1),)
    prop = ZeroPadPropagator(pad_width)
    _ = prop @ wave

    ua = UA.zeros((3,), batch_size=1, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 7.0

    out = prop._compute_forward({"input": ua})
    assert out.data.shape == (1, 6)  # 3 + 2 + 1 = 6
    assert xp.all(out.data[:, :2] == 0)
    assert xp.all(out.data[:, -1] == 0)
    assert xp.all(out.data[:, 2:5] == 7.0)

@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output_zero_pad(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), batch_size=1, dtype=xp.complex64)
    sample = xp.arange(4, dtype=xp.complex64).reshape(1, 2, 2)
    wave.set_sample(sample)

    prop = ZeroPadPropagator(((1, 1), (2, 0)))
    _ = prop @ wave

    padded = prop.get_sample_for_output()
    assert padded.shape == (1, 4, 4)   # (2+1+1, 2+2+0)
    assert xp.allclose(padded[:, 1:3, 2:4], sample)

@pytest.mark.parametrize("xp", backend_libs)
def test_wave_zero_pad_get_sample(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), batch_size=1, dtype=xp.complex64)
    sample = xp.arange(4, dtype=xp.complex64).reshape(1, 2, 2)
    wave.set_sample(sample)

    pad_width = ((1, 0), (0, 2))
    out = wave.zero_pad(pad_width)

    padded = out.parent.get_sample_for_output()
    assert padded.shape == (1, 3, 4)

    # 中央部分が元の sample に一致する
    assert xp.allclose(padded[:, 1:, 0:2], sample)

