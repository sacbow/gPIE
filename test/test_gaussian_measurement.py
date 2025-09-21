import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
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
def test_basic_set_observed(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.float32)
    meas = GaussianMeasurement(var=0.5) << wave

    obs = xp.ones((1, 2, 2), dtype=xp.float32)
    meas.set_observed(obs)

    assert isinstance(meas.observed, UncertainArray)
    assert meas.observed.dtype == xp.float32
    assert meas.observed.event_shape == (2, 2)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_with_mask_and_array_precision(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.float64)
    meas = GaussianMeasurement(var=2.0, precision_mode="array") << wave

    data = xp.ones((1, 2, 2), dtype=xp.float64)
    mask = xp.array([[[1, 0], [0, 1]]], dtype=bool)
    meas.set_observed(data, mask=mask)

    prec = meas.observed.precision(raw=True)
    assert xp.allclose(prec[mask], 1.0 / 2.0)
    assert xp.all(prec[~mask] == 0.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_generate_sample_and_promote_to_observed(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.float32)
    wave.set_sample(xp.zeros((1, 2, 2), dtype=xp.float32))
    meas = GaussianMeasurement(var=0.1) << wave

    rng = get_rng(seed=123)
    meas._generate_sample(rng)
    meas.update_observed_from_sample()

    assert isinstance(meas.observed, UncertainArray)
    assert meas.observed.event_shape == (2, 2)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_dtype_cast(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,), dtype=xp.float32)
    meas = GaussianMeasurement(var=0.1) << wave

    obs = xp.array([[1.0, 2.0]], dtype=xp.float32)
    meas.set_observed(obs)

    assert meas.observed.dtype == xp.float32


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_invalid_dtype_raises(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,), dtype=xp.float32)
    meas = GaussianMeasurement(var=0.1) << wave

    obs = xp.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=xp.complex64)
    with pytest.raises(TypeError):
        meas.set_observed(obs)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_batched_false(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,), dtype=xp.float32)
    meas = GaussianMeasurement(var=0.1) << wave

    obs = xp.array([1.0, 2.0], dtype=xp.float32)
    meas.set_observed(obs, batched=False)

    assert meas.observed.batch_shape == (1,)
    assert meas.observed.event_shape == (2,)
