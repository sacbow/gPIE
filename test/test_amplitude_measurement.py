import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.measurement.amplitude_measurement import AmplitudeMeasurement
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.rng_utils import get_rng
from gpie.core.types import PrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_dtype_inference_from_input_dtype(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2) << wave
    assert m.input_dtype == xp.dtype(xp.complex64)
    assert m.observed_dtype == xp.dtype(xp.float32)


@pytest.mark.parametrize("xp", backend_libs)
def test_generate_and_update_observed(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex128)
    wave.set_sample(xp.ones((1, 2, 2), dtype=xp.complex128))
    m = AmplitudeMeasurement(var=1e-3) << wave

    rng = get_rng(seed=42)
    m._generate_sample(rng)
    assert m.get_sample() is not None

    m.update_observed_from_sample()
    assert isinstance(m.observed, UncertainArray)
    assert m.observed.batch_shape == (1,)
    assert m.observed.event_shape == (2, 2)
    assert np.issubdtype(m.observed.dtype, np.floating)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_with_mask_scalar_precision(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    wave.set_sample(xp.ones((1, 2, 2), dtype=xp.complex64))
    m = AmplitudeMeasurement(var=0.5, precision_mode="scalar") << wave

    data = xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32).reshape(1, 2, 2)
    mask = xp.array([[True, False], [True, False]], dtype=bool).reshape(1, 2, 2)
    m.set_observed(data, mask=mask)
    assert isinstance(m.observed, UncertainArray)
    assert m.observed.batch_shape == (1,)
    assert m.observed.event_shape == (2, 2)
    assert np.allclose(m.observed.precision(raw=True)[mask], 2.0)
    assert np.allclose(m.observed.precision(raw=True)[~mask], 0.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_invalid_dtype_raises(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,), dtype=xp.complex64)
    meas = AmplitudeMeasurement(var=0.1) << wave

    obs = xp.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=xp.complex64)
    with pytest.raises(TypeError):
        meas.set_observed(obs)


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_message_and_shapes(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    wave.set_sample(xp.ones((1, 2, 2), dtype=xp.complex64))
    
    m = AmplitudeMeasurement(var=1e-2, precision_mode="scalar") << wave

    m._generate_sample(get_rng(seed=0))
    m.update_observed_from_sample()

    incoming = UncertainArray(wave.get_sample(), dtype=xp.complex64, precision=1.0, batched=True)
    msg = m._compute_message(incoming)
    assert isinstance(msg, UncertainArray)
    assert msg.batch_shape == (1,)
    assert msg.event_shape == (2, 2)
    assert np.issubdtype(msg.dtype, xp.complexfloating)

