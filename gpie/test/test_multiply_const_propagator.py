import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.multiply_const_propagator import MultiplyConstPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.rng_utils import get_rng
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("const_val", [2.0, np.array([[2.0]]), np.array([[2.0, 2.0], [2.0, 2.0]])])
def test_forward_backward_scalar_and_uniform(xp, const_val):
    """Test MultiplyConstPropagator forward/backward with scalar and uniform array constants."""
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    shape = (2, 2)
    wave = Wave(shape=shape, dtype=xp.complex64)
    ua_in = UA.random(shape, dtype=xp.complex64, rng=rng, scalar_precision=True)

    prop = MultiplyConstPropagator(const=const_val)
    output = prop @ wave

    # Inject input message and test forward
    prop.input_messages[wave] = ua_in
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
    msg_out = prop._compute_forward({"input": ua_in})
    assert msg_out.data.shape == shape
    assert xp.allclose(msg_out.data, ua_in.data * xp.asarray(prop.const_safe))

    # Backward: test precision scaling
    msg_back = prop._compute_backward(msg_out, exclude="input")
    assert msg_back.data.shape == shape
    assert xp.allclose(msg_back.data, msg_out.data / xp.asarray(prop.const_safe))


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_non_uniform_precision_mode(xp):
    """Test SCALAR_TO_ARRAY precision mode when constant magnitudes are non-uniform."""
    backend.set_backend(xp)
    rng = get_rng(seed=1)

    shape = (2, 2)
    wave = Wave(shape=shape, dtype=xp.complex64)
    const = xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=xp.complex64)  # non-uniform magnitude
    prop = MultiplyConstPropagator(const=const)
    output = prop @ wave

    # SCALAR_TO_ARRAY precision test
    ua_in = UA.random(shape, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[wave] = ua_in
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    msg_out = prop._compute_forward({"input": ua_in})
    assert msg_out.precision(raw = True).shape == shape  # array precision

    msg_back = prop._compute_backward(msg_out, exclude="input")
    assert msg_back.data.shape == shape


@pytest.mark.parametrize("xp", backend_libs)
def test_dtype_lowering_and_broadcast(xp):
    """Test dtype lowering (complex128 + complex64 → complex64) and broadcasting behavior."""
    backend.set_backend(xp)
    shape = (2, 2)
    wave = Wave(shape=shape, dtype=xp.complex64)
    const = xp.array([2.0], dtype=xp.complex128)  # higher precision const

    prop = MultiplyConstPropagator(const=const)
    output = prop @ wave
    assert output.dtype == xp.complex64  # lowered to wave dtype

    # Broadcasting success
    assert prop.const.shape == shape

    # Broadcasting failure
    bad_const = xp.ones((3, 3), dtype=xp.complex64)
    prop_bad = MultiplyConstPropagator(bad_const)
    with pytest.raises(ValueError):
        _ = prop_bad @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_transfer(xp):
    """Test to_backend properly transfers constants and related arrays."""
    backend.set_backend(xp)
    const = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.complex128)
    prop = MultiplyConstPropagator(const=const)

    # Switch backend
    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.const, new_backend.ndarray)
    assert isinstance(prop.const_safe, new_backend.ndarray)
    assert isinstance(prop.inv_amp_sq, new_backend.ndarray)


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_with_rng_init(xp):
    """Test forward initialization path with RNG before messages are set."""
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    shape = (2, 2)
    wave = Wave(shape=shape, dtype=xp.complex64)
    prop = MultiplyConstPropagator(const=2.0)
    output = prop @ wave

    prop.set_init_rng(rng)
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
    # No input/output message yet, triggers random init forward
    prop.forward()
    assert isinstance(output.parent_message, UA)


@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output(xp):
    """Test sample generation multiplies input sample by constant."""
    backend.set_backend(xp)
    shape = (2, 2)
    wave = Wave(shape=shape, dtype=xp.complex64)
    sample = xp.ones(shape, dtype=xp.complex64)
    wave.set_sample(sample)

    prop = MultiplyConstPropagator(const=2.0)
    output = prop @ wave
    out_sample = prop.get_sample_for_output()
    assert xp.allclose(out_sample, sample * 2.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_repr(xp):
    """Test __repr__ outputs mode and shape/scalar info."""
    backend.set_backend(xp)
    prop_scalar = MultiplyConstPropagator(2.0)
    prop_array = MultiplyConstPropagator(xp.ones((2, 2)))
    assert "scalar" in repr(prop_scalar)
    assert "shape" in repr(prop_array)
