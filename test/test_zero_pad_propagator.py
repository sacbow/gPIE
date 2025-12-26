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

@pytest.mark.parametrize("xp", backend_libs)
def test_zero_pad_blockwise_forward_backward(xp):
    backend.set_backend(xp)

    B = 4
    H, W = 2, 3
    block = slice(1, 3)

    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    pad_width = ((1, 1), (2, 0))
    prop = ZeroPadPropagator(pad_width)
    _ = prop @ wave

    # Input UA
    ua = UA.zeros((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    for b in range(B):
        ua.data[b] = b + 1  # make batch elements distinguishable

    # -------------------------
    # Forward: full vs block
    # -------------------------
    out_full = prop._compute_forward({"input": ua})
    out_blk = prop._compute_forward({"input": ua}, block=block)

    assert out_blk.batch_size == block.stop - block.start
    assert xp.allclose(
        out_blk.data,
        out_full.data[block],
    )

    # -------------------------
    # Backward: full vs block
    # -------------------------
    back_full = prop._compute_backward(out_full, exclude="input")
    back_blk = prop._compute_backward(out_full, exclude="input", block=block)

    assert back_blk.batch_size == block.stop - block.start
    assert xp.allclose(
        back_blk.data,
        back_full.data[block],
    )

import importlib.util
import pytest
import numpy as np

import gpie
from gpie import model, fft2, mse, Graph
from gpie import SparsePrior, GaussianMeasurement
from gpie.core.linalg_utils import random_binary_mask
from gpie.core.rng_utils import get_rng

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


def build_fft_cs_graph(
    *,
    xp,
    event_shape,
    batch_size,
    rho,
    var,
    mask,
    seed_init=11,
    seed_sample=123,
):
    """
    Build and initialize an FFT-based compressive sensing graph
    using the @model decorator.
    """

    @model
    def fft_cs_model(rho, shape, var, batch_size):
        x = ~SparsePrior(
            rho=rho,
            event_shape=shape,
            batch_size=batch_size,
            label="x",
            dtype=xp.complex64,
        )
        GaussianMeasurement(var=var, with_mask=True) << fft2(x)

    g = fft_cs_model(
        rho=rho,
        shape=event_shape,
        var=var,
        batch_size=batch_size,
    )

    # Initialization & sampling
    g.set_init_rng(get_rng(seed=seed_init))
    g.generate_sample(
        rng=get_rng(seed=seed_sample),
        update_observed=True,
        mask=mask,
    )

    return g

