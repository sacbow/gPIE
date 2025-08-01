import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.add_propagator import AddPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.rng_utils import get_rng
from gpie.core.types import get_lower_precision_dtype, BinaryPropagatorPrecisionMode as BPM

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_and_dtype(xp):
    """Test AddPropagator forward/backward computations and dtype lowering."""
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    a_wave = Wave(shape=(2, 2), dtype=xp.complex64)
    b_wave = Wave(shape=(2, 2), dtype=xp.complex128)
    out_wave = AddPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    # Generate input messages
    ua_a = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision=True)
    ua_b = UA.random((2, 2), dtype=xp.complex128, rng=rng, scalar_precision=True)
    prop.input_messages[a_wave] = ua_a
    prop.input_messages[b_wave] = ua_b
    prop.dtype = get_lower_precision_dtype(ua_a.dtype, ua_b.dtype)

    # Forward pass: fused mean and harmonic precision
    ua_out = prop._compute_forward({"a": ua_a, "b": ua_b})
    assert ua_out.dtype == prop.dtype
    expected_mu = ua_a.data + ua_b.data
    expected_prec = 1.0 / (1.0 / ua_a.precision(raw=True) + 1.0 / ua_b.precision(raw=True))
    assert xp.allclose(ua_out.data, expected_mu)
    assert xp.allclose(ua_out.precision(raw=True), expected_prec)

    # Backward pass: residual message to "a"
    ua_back_a = prop._compute_backward(ua_out, exclude="a")
    assert xp.allclose(ua_back_a.data, ua_out.data - ua_b.data)

    # Backward pass: residual message to "b"
    ua_back_b = prop._compute_backward(ua_out, exclude="b")
    assert xp.allclose(ua_back_b.data, ua_out.data - ua_a.data)


@pytest.mark.parametrize("xp", backend_libs)
def test_backward_with_mixed_precision_projection(xp):
    """Test backward projection for mixed precision modes."""
    backend.set_backend(xp)
    rng = get_rng(seed=1)

    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    out_wave = AddPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    ua_a = UA.random((2, 2), dtype=xp.float32, rng=rng, scalar_precision=True)
    ua_b = UA.random((2, 2), dtype=xp.float32, rng=rng, scalar_precision=False)
    ua_out = UA.random((2, 2), dtype=xp.float32, rng=rng, scalar_precision=False)

    prop.input_messages[a_wave] = ua_a
    prop.input_messages[b_wave] = ua_b

    # Test SCALAR_AND_ARRAY_TO_ARRAY (exclude "a")
    prop._precision_mode = BPM.SCALAR_AND_ARRAY_TO_ARRAY
    msg_a = prop._compute_backward(ua_out, exclude="a")
    assert isinstance(msg_a, UA)
    assert msg_a._scalar_precision is True  # scalar precision

    # (exclude "b")
    msg_b = prop._compute_backward(ua_out, exclude="b")
    assert isinstance(msg_b, UA)
    assert msg_b._scalar_precision is False  # scalar precision


@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output_and_repr(xp):
    """Test sample generation and __repr__ formatting."""
    backend.set_backend(xp)
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    a_wave.set_sample(xp.ones((2, 2), dtype=xp.float32))
    b_wave.set_sample(2 * xp.ones((2, 2), dtype=xp.float32))

    out_wave = AddPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    out_sample = prop.get_sample_for_output(rng=get_rng(seed=0))
    assert xp.allclose(out_sample, 3.0)

    rep = repr(prop)
    assert "Add" in rep
    assert "mode" in rep


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_missing_inputs(xp):
    """Test runtime errors when inputs/messages are missing."""
    backend.set_backend(xp)
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    out_wave = AddPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    # Missing message raises
    with pytest.raises(RuntimeError):
        prop._compute_forward({"a": None, "b": None})

    # Missing wave in backward raises
    with pytest.raises(RuntimeError):
        prop._compute_backward(UA.random((2, 2), dtype=xp.float32, rng=get_rng()), exclude="c")

    # Missing input message in backward raises
    with pytest.raises(RuntimeError):
        prop._compute_backward(UA.random((2, 2), dtype=xp.float32, rng=get_rng()), exclude="a")
