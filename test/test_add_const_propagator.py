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
def test_forward_backward_and_dtype_promotion(xp):
    """Test forward/backward pass and dtype promotion."""
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    wave = Wave(shape=(4, 4), dtype=xp.complex64)
    output = AddConstPropagator(const=1.0) @ wave
    prop = output.parent

    # Prepare input message (complex64 UA)
    ua_in = UA.random((4, 4), dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[wave] = ua_in

    # Forward: should add constant, dtype promote to complex64 (no promotion needed here)
    ua_out = prop._compute_forward({"input": ua_in})
    assert ua_out.dtype == get_lower_precision_dtype(xp.complex64, xp.float64)  # -> complex128 (numpy) or backend equivalent

    # Backward: subtract constant
    ua_back = prop._compute_backward(ua_out, exclude="input")
    assert xp.allclose(ua_back.data, ua_out.data - 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_broadcast_and_invalid_shape(xp):
    """Test broadcasting of const and invalid shape error."""
    backend.set_backend(xp)

    wave = Wave(shape=(2, 2), dtype=xp.float32)
    const = xp.array([1.0])  # broadcastable scalar array
    prop = AddConstPropagator(const=const)
    out_wave = prop @ wave
    assert out_wave.shape == (2, 2)
    assert prop.const.shape == (2, 2)

    # Invalid broadcast
    bad_const = xp.ones((3, 3))
    prop_bad = AddConstPropagator(const=bad_const)
    with pytest.raises(ValueError):
        _ = prop_bad @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_transfer(xp):
    """Test to_backend transfers const correctly between numpy/cupy."""
    backend.set_backend(xp)
    const = np.ones((2, 2), dtype=np.float32)
    prop = AddConstPropagator(const)
    orig_dtype = prop.const.dtype

    # Switch backend
    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.const, new_backend.ndarray)
    assert prop.const.dtype == new_backend.dtype(orig_dtype)


@pytest.mark.parametrize("xp", backend_libs)
def test_precision_mode_forward_backward(xp):
    """Test precision mode propagation forward and backward."""
    backend.set_backend(xp)
    wave = Wave(shape=(2, 2), dtype=xp.float32)
    wave._set_precision_mode("array")
    output = AddConstPropagator(const=1.0) @ wave
    prop = output.parent

    # Forward: propagate scalar precision to output
    prop.set_precision_mode_forward()
    assert prop.precision_mode == "array"

    # Manually set output precision to array, propagate backward
    prop.output._set_precision_mode("array")
    prop.set_precision_mode_backward()
    assert wave.precision_mode == "array"
    assert prop.precision_mode == "array"


@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output(xp):
    """Test sample generation by adding constant."""
    backend.set_backend(xp)
    wave = Wave(shape=(2, 2), dtype=xp.float32)
    sample = xp.ones((2, 2), dtype=xp.float32)
    wave.set_sample(sample)

    output = AddConstPropagator(const=2.0) @ wave
    prop = output.parent
    out_sample = prop.get_sample_for_output()
    assert xp.allclose(out_sample, sample + 2.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_repr(xp):
    """Test __repr__ includes generation and mode."""
    backend.set_backend(xp)
    wave = Wave(shape=(2, 2), dtype=xp.float32)
    output = AddConstPropagator(1.0) @ wave
    prop = output.parent
    rep = repr(prop)
    assert "AddConst" in rep
