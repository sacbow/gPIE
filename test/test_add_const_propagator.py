import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.add_const_propagator import AddConstPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.rng_utils import get_rng
from gpie.core.types import get_lower_precision_dtype

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_add_const_forward_backward_dtype(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    B, H, W = 3, 4, 4

    x = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    output = AddConstPropagator(const=1.0) @ x
    prop = output.parent

    ua_in = UA.random((H, W), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[x] = ua_in

    ua_out = prop._compute_forward({"input": ua_in})
    expected_dtype = get_lower_precision_dtype(xp.complex64, xp.float64)
    assert ua_out.dtype == expected_dtype

    ua_back = prop._compute_backward(ua_out, exclude="input")
    assert xp.allclose(ua_back.data, ua_out.data - 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_add_const_broadcast_and_shape_mismatch(xp):
    backend.set_backend(xp)
    B, H, W = 2, 3, 3

    # Valid 2D broadcast
    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.float32)
    const = xp.ones((H, W), dtype=xp.float32)
    prop = AddConstPropagator(const=const)
    out = prop @ wave
    assert out.event_shape == (H, W)
    assert out.batch_size == B
    assert prop.const.shape == (B, H, W)

    # Invalid: shape mismatch
    bad_const = xp.ones((4, 4), dtype=xp.float32)
    with pytest.raises(ValueError):
        _ = AddConstPropagator(bad_const) @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_add_const_to_backend(xp):
    backend.set_backend(xp)
    const = np.ones((2, 2), dtype=np.float32)
    prop = AddConstPropagator(const)

    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.const, new_backend.ndarray)
    assert prop.const.dtype == new_backend.dtype(np.float32)


@pytest.mark.parametrize("xp", backend_libs)
def test_add_const_precision_forward_backward(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), batch_size=2, dtype=xp.float32)
    wave._set_precision_mode("array")
    output = AddConstPropagator(const=1.0) @ wave
    prop = output.parent

    prop.set_precision_mode_forward()
    assert prop.precision_mode == "array"

    prop.output._set_precision_mode("array")
    prop.set_precision_mode_backward()
    assert wave.precision_mode == "array"
    assert prop.precision_mode == "array"


@pytest.mark.parametrize("xp", backend_libs)
def test_add_const_get_sample_for_output(xp):
    backend.set_backend(xp)
    B, H, W = 3, 2, 2
    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.float32)
    sample = xp.ones((B, H, W), dtype=xp.float32)
    wave.set_sample(sample)

    output = AddConstPropagator(const=2.0) @ wave
    prop = output.parent
    out_sample = prop.get_sample_for_output()
    assert xp.allclose(out_sample, sample + 2.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_add_const_repr(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), batch_size=2, dtype=xp.float32)
    output = AddConstPropagator(1.0) @ wave
    prop = output.parent
    rep = repr(prop)
    assert "AddConst" in rep
    assert "mode=" in rep

@pytest.mark.parametrize("xp", backend_libs)
def test_add_const_blockwise_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    B, H, W = 4, 3, 3
    block = slice(1, 3)

    # Create wave and propagator
    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.float32)
    output = AddConstPropagator(const=2.0) @ wave
    prop = output.parent

    # Input UA
    ua_in = UA.random(
        (H, W),
        batch_size=B,
        dtype=xp.float32,
        rng=rng,
        scalar_precision=True,
    )

    # -------------------------------
    # Forward: full vs block-wise
    # -------------------------------
    ua_full = prop._compute_forward({"input": ua_in})
    ua_blk = prop._compute_forward({"input": ua_in}, block=block)

    # Block output must match sliced full output
    assert ua_blk.batch_size == block.stop - block.start
    assert xp.allclose(
        ua_blk.data,
        ua_full.data[block],
    )

    # Precision must be preserved
    assert xp.allclose(
        ua_blk.precision(raw=True),
        ua_full.precision(raw=True)[block],
    )

    # -------------------------------
    # Backward: full vs block-wise
    # -------------------------------
    ua_back_full = prop._compute_backward(ua_full, exclude="input")
    ua_back_blk = prop._compute_backward(ua_full, exclude="input", block=block)

    assert ua_back_blk.batch_size == block.stop - block.start
    assert xp.allclose(
        ua_back_blk.data,
        ua_back_full.data[block],
    )

    assert xp.allclose(
        ua_back_blk.precision(raw=True),
        ua_back_full.precision(raw=True)[block],
    )
