import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.multiply_const_propagator import MultiplyConstPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.rng_utils import get_rng
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_real_dtype

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("const_val", [
    2.0,
    np.array([[2.0, 2.0], [2.0, 2.0]]),
    np.array([[[2.0, 2.0], [2.0, 2.0]]])
])
def test_forward_backward_scalar_and_uniform(xp, const_val):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    shape = (2, 2)
    B = 1
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)
    ua_in = UA.random(event_shape=shape, dtype=xp.complex64, rng=rng, scalar_precision=True)

    prop = MultiplyConstPropagator(const=const_val)
    output = prop @ wave

    prop.input_messages[wave] = ua_in
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    msg_out = prop._compute_forward(ua_in)
    assert msg_out.event_shape == shape

    abs_sq = xp.abs(prop.const) ** 2
    eps = xp.array(prop._eps, dtype=get_real_dtype(prop.const_dtype))

    expected_mu_out = ua_in.data * prop.const
    expected_prec_out = ua_in.precision(raw=True) / (abs_sq + eps)
    assert xp.allclose(msg_out.data, expected_mu_out)
    assert xp.allclose(msg_out.precision(raw=True), expected_prec_out)

    msg_back = prop._compute_backward(msg_out)
    expected_mu_in = msg_out.data * xp.conj(prop.const) / (abs_sq + eps)
    expected_prec_in = msg_out.precision(raw=True) * abs_sq
    assert msg_back.event_shape == shape
    assert xp.allclose(msg_back.data, expected_mu_in)
    assert xp.allclose(msg_back.precision(raw=True), expected_prec_in)


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_non_uniform_precision_mode(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=1)
    shape = (2, 2)
    B = 1
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)
    const = xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=xp.complex64)
    prop = MultiplyConstPropagator(const=const)
    output = prop @ wave

    ua_in = UA.random(shape, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[wave] = ua_in
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    msg_out = prop._compute_forward(ua_in)
    assert msg_out.precision(raw=True).shape == (B, *shape)

    msg_back = prop._compute_backward(msg_out)
    assert msg_back.event_shape == shape


@pytest.mark.parametrize("xp", backend_libs)
def test_dtype_lowering_and_broadcast(xp):
    backend.set_backend(xp)
    shape = (2, 2)
    B = 1
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)

    # High precision const (will be cast to lower precision)
    const = xp.array([2.0], dtype=xp.complex128)
    prop = MultiplyConstPropagator(const=const)
    output = prop @ wave

    assert output.dtype == xp.complex64
    assert prop.const.shape == (B, *shape)

    # Broadcasting error
    bad_const = xp.ones((3, 3), dtype=xp.complex64)
    prop_bad = MultiplyConstPropagator(bad_const)
    with pytest.raises(ValueError):
        _ = prop_bad @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_transfer(xp):
    backend.set_backend(xp)
    const = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.complex128)
    prop = MultiplyConstPropagator(const=const)

    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.const, new_backend.ndarray)
    assert isinstance(prop.const_conj, new_backend.ndarray)
    assert isinstance(prop.const_abs_sq, new_backend.ndarray)
    assert isinstance(prop._eps, new_backend.ndarray)


@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output(xp):
    backend.set_backend(xp)
    shape = (2, 2)
    B = 1
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)
    sample = xp.ones((B, *shape), dtype=xp.complex64)
    wave.set_sample(sample)

    prop = MultiplyConstPropagator(const=2.0)
    output = prop @ wave
    out_sample = prop.get_sample_for_output()
    assert xp.allclose(out_sample, sample * 2.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_repr(xp):
    backend.set_backend(xp)
    prop_scalar = MultiplyConstPropagator(2.0)
    prop_array = MultiplyConstPropagator(xp.ones((2, 2)))
    assert "scalar" in repr(prop_scalar)
    assert "shape" in repr(prop_array)


@pytest.mark.parametrize("xp", [np])
def test_set_precision_mode_conflicts_and_invalid(xp):
    backend.set_backend(xp)
    prop = MultiplyConstPropagator(2.0)

    # invalid string
    with pytest.raises(ValueError):
        prop._set_precision_mode("invalid")

    # set once, then conflict
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
    with pytest.raises(ValueError):
        prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)


@pytest.mark.parametrize("xp", [np])
def test_set_precision_mode_forward_and_backward(xp):
    backend.set_backend(xp)
    shape = (2, 2)
    B = 1

    # forward: input wave SCALAR → SCALAR_TO_ARRAY
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)
    output = MultiplyConstPropagator(2.0) @ wave
    prop = output.parent
    wave._set_precision_mode(PrecisionMode.SCALAR)
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY

    # backward: output SCALAR → ARRAY_TO_SCALAR
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)
    output = MultiplyConstPropagator(2.0) @ wave
    prop = output.parent
    prop.output._set_precision_mode(PrecisionMode.SCALAR)
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR

    # backward: input SCALAR → SCALAR_TO_ARRAY
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)
    output = MultiplyConstPropagator(2.0) @ wave
    prop = output.parent
    wave._set_precision_mode(PrecisionMode.SCALAR)
    prop.output._set_precision_mode(PrecisionMode.ARRAY)
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY

    # backward: both ARRAY → ARRAY
    wave = Wave(event_shape=shape, batch_size=B, dtype=xp.complex64)
    output = MultiplyConstPropagator(2.0) @ wave
    prop = output.parent
    wave._set_precision_mode(PrecisionMode.ARRAY)
    prop.output._set_precision_mode(PrecisionMode.ARRAY)
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY


@pytest.mark.parametrize("xp", [np])
def test_get_input_output_precision_modes(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), batch_size=1, dtype=xp.complex64)
    output = MultiplyConstPropagator(2.0) @ wave
    prop = output.parent

    # none
    assert prop.get_input_precision_mode(wave) is None
    assert prop.get_output_precision_mode() is None

    # SCALAR_TO_ARRAY
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
    assert prop.get_input_precision_mode(wave) == PrecisionMode.SCALAR
    assert prop.get_output_precision_mode() == PrecisionMode.ARRAY

    # ARRAY_TO_SCALAR
    wave = Wave(event_shape=(2, 2), batch_size=1, dtype=xp.complex64)
    output = MultiplyConstPropagator(2.0) @ wave
    prop = output.parent
    prop._precision_mode = UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR
    assert prop.get_input_precision_mode(wave) == PrecisionMode.ARRAY
    assert prop.get_output_precision_mode() == PrecisionMode.SCALAR
