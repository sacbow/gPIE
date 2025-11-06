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


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_belief_and_compute_fitness(xp):
    """Check GaussianMeasurement.compute_belief() and compute_fitness()."""
    backend.set_backend(xp)

    # Create Wave and connect Measurement
    wave = Wave(event_shape=(2, 2), dtype=xp.float32)
    meas = GaussianMeasurement(var=0.1) << wave

    # Create deterministic message and observation with batch dimension
    msg_mean = xp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=xp.float32)  # shape (1,2,2)
    msg_prec = xp.ones_like(msg_mean) * 2.0
    obs_data = xp.array([[[1.1, 2.1], [2.9, 4.1]]], dtype=xp.float32)
    obs_prec = xp.ones_like(obs_data) * 10.0

    # Assign message and observation
    from gpie.core.uncertain_array import UncertainArray as UA
    ua_msg = UA(msg_mean, dtype=xp.float32, precision=msg_prec, batched=True)
    meas.input_messages[wave] = ua_msg
    meas.set_observed(obs_data, precision=obs_prec)

    # --- test compute_belief ---
    belief = meas.compute_belief()
    assert isinstance(belief, UA)
    gamma_expected = msg_prec + obs_prec
    mu_expected = (msg_prec * msg_mean + obs_prec * obs_data) / gamma_expected
    assert xp.allclose(belief.data, mu_expected)
    assert xp.allclose(belief.precision(raw=True), gamma_expected)

    # --- test compute_fitness ---
    f = meas.compute_fitness()
    diff2 = xp.abs(mu_expected - obs_data) ** 2
    weighted = xp.mean(obs_prec * diff2)
    assert np.allclose(f, float(weighted), atol=1e-7)



@pytest.mark.parametrize("xp", backend_libs)
def test_compute_fitness_with_mask(xp):
    """Check that masked elements (precision=0) do not contribute to fitness."""
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.float32)
    meas = GaussianMeasurement(var=1.0, precision_mode="array", with_mask = True) << wave

    msg_mean = xp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=xp.float32)
    msg_prec = xp.ones_like(msg_mean)
    obs_data = xp.array([[[1.0, 2.0], [5.0, 0.0]]], dtype=xp.float32)
    mask = xp.array([[[1, 0], [1, 0]]], dtype=bool)  # shape (1,2,2)

    from gpie.core.uncertain_array import UncertainArray as UA
    ua_msg = UA(msg_mean, dtype=xp.float32, precision=msg_prec, batched=True)
    meas.input_messages[wave] = ua_msg
    meas.set_observed(obs_data, mask=mask)

    f = meas.compute_fitness()

    gamma = meas.observed.precision(raw=True)
    assert xp.all(gamma[~mask] == 0.0)

