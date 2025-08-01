import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.multiply_propagator import MultiplyPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.rng_utils import get_rng
from gpie.core.types import PrecisionMode, BinaryPropagatorPrecisionMode as BPMM, get_lower_precision_dtype

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_moment_matching(xp):
    """Test _compute_forward with belief-based moment matching."""
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    # Prepare input waves
    a_wave = Wave((2, 2), dtype=xp.complex64)
    b_wave = Wave((2, 2), dtype=xp.complex64)
    output = MultiplyPropagator() @ (a_wave, b_wave)
    prop = output.parent

    # Beliefs for inputs
    ua_a = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)
    ua_b = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)
    a_wave.set_belief(ua_a)
    b_wave.set_belief(ua_b)
    prop.dtype = get_lower_precision_dtype(ua_a.dtype, ua_b.dtype)

    # Forward compute
    ua_out = prop._compute_forward({"a": ua_a, "b": ua_b})
    assert ua_out.dtype == prop.dtype
    assert ua_out.data.shape == (2, 2)
    assert xp.all(ua_out.precision(raw=True) > 0)  # precision positive


@pytest.mark.parametrize("xp", backend_libs)
def test_backward_message_and_belief_update(xp):
    """Test _compute_backward updates message and belief."""
    backend.set_backend(xp)
    rng = get_rng(seed=1)

    a_wave = Wave((2, 2), dtype=xp.complex64)
    b_wave = Wave((2, 2), dtype=xp.complex64)
    out_wave = MultiplyPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    # Assign beliefs and messages
    ua_a = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision = False)
    ua_b = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision = False)
    ua_out = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision = False)

    a_wave.set_belief(ua_a)
    b_wave.set_belief(ua_b)
    prop.input_messages[a_wave] = ua_a
    prop.input_messages[b_wave] = ua_b

    # Backward for exclude='a'
    msg_a, belief_a = prop._compute_backward(ua_out, exclude="a")
    assert isinstance(msg_a, UA)
    assert isinstance(belief_a, UA)
    assert msg_a.data.shape == (2, 2)
    assert belief_a.data.shape == (2, 2)


@pytest.mark.parametrize("xp", backend_libs)
def test_precision_modes(xp):
    """Test precision mode selection rules."""
    backend.set_backend(xp)
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)

    prop = MultiplyPropagator() @ (a_wave, b_wave)

    # scalar × scalar -> ValueError
    a_wave._set_precision_mode("scalar")
    b_wave._set_precision_mode("scalar")
    with pytest.raises(ValueError):
        prop.parent.set_precision_mode_forward()

    # scalar × array -> SCALAR_AND_ARRAY_TO_ARRAY
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    prop = MultiplyPropagator() @ (a_wave, b_wave)
    a_wave._set_precision_mode("scalar")
    b_wave._set_precision_mode("array")
    prop.parent.set_precision_mode_forward()
    assert prop.parent.precision_mode_enum == BPMM.SCALAR_AND_ARRAY_TO_ARRAY

    # array × scalar -> ARRAY_AND_SCALAR_TO_ARRAY
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    prop = MultiplyPropagator() @ (a_wave, b_wave)
    a_wave._set_precision_mode("array")
    b_wave._set_precision_mode("scalar")
    prop.parent.set_precision_mode_forward()
    assert prop.parent.precision_mode_enum == BPMM.ARRAY_AND_SCALAR_TO_ARRAY

    # array × array -> ARRAY
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    prop = MultiplyPropagator() @ (a_wave, b_wave)
    a_wave._set_precision_mode("array")
    b_wave._set_precision_mode("array")
    prop.parent.set_precision_mode_forward()
    assert prop.parent.precision_mode_enum == BPMM.ARRAY


@pytest.mark.parametrize("xp", backend_libs)
def test_dtype_lowering(xp):
    """Test dtype lowering rules in MultiplyPropagator."""
    backend.set_backend(xp)
    a_wave = Wave((2, 2), dtype=xp.complex64)
    b_wave = Wave((2, 2), dtype=xp.complex128)
    out_wave = MultiplyPropagator() @ (a_wave, b_wave)
    assert out_wave.dtype == get_lower_precision_dtype(xp.complex64, xp.complex128)


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_errors(xp):
    """Test error cases in forward/backward."""
    backend.set_backend(xp)
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    out_wave = MultiplyPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    # forward: missing belief
    with pytest.raises(RuntimeError):
        prop._compute_forward({"a": None, "b": None})

    # backward: missing output message
    with pytest.raises(RuntimeError):
        prop.backward()

    # generate_sample: missing input sample
    with pytest.raises(RuntimeError):
        prop.generate_sample(rng=get_rng())


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_with_rng_init(xp):
    """Test forward initializes random UA if beliefs missing."""
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    out_wave = MultiplyPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    prop.set_init_rng(rng)
    prop.forward()  # should not raise, sends random UA
    assert isinstance(out_wave.parent_message, UA)


@pytest.mark.parametrize("xp", backend_libs)
def test_repr_and_sample(xp):
    """Test __repr__ and sample generation."""
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    a_wave = Wave((2, 2), dtype=xp.float32)
    b_wave = Wave((2, 2), dtype=xp.float32)
    out_wave = MultiplyPropagator() @ (a_wave, b_wave)
    prop = out_wave.parent

    # set input samples
    a_wave.set_sample(xp.ones((2, 2), dtype=xp.float32))
    b_wave.set_sample(2 * xp.ones((2, 2), dtype=xp.float32))
    prop.generate_sample(rng)
    assert xp.allclose(out_wave.get_sample(), 2.0)

    # repr
    rep = repr(prop)
    assert "Mul" in rep
    assert "mode" in rep
