import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.measurement.amplitude_measurement import AmplitudeMeasurement
from gpie.graph.wave import Wave
from gpie.graph.structure.graph import Graph
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
def test_amplitude_measurement_init_and_dtype_inference(xp):
    backend.set_backend(xp)
    obs = xp.ones((4, 4), dtype=xp.float32)
    meas = AmplitudeMeasurement(observed_data=obs, var=1e-3)
    assert meas.observed is not None
    assert meas.observed.data.dtype == xp.float32  
    assert np.issubdtype(meas.expected_observed_dtype, np.floating)


@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_measurement_incompatible_observed_dtype(xp):
    backend.set_backend(xp)
    obs = xp.ones((4, 4), dtype=xp.complex64)
    with pytest.raises(TypeError):
        AmplitudeMeasurement(observed_data=obs, var=1e-3)


@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_measurement_mask_and_precision(xp):
    backend.set_backend(xp)
    obs = xp.ones((3, 3), dtype=xp.float32)
    mask = xp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    meas = AmplitudeMeasurement(observed_data=obs, var=1e-2, mask=mask)
    assert meas.mask is not None
    prec = meas.observed.precision(raw=True)
    assert xp.allclose(prec[mask], 1.0 / 1e-2)
    assert xp.all(prec[~mask] == 0.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_measurement_generate_and_update_observed(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(4, 4), dtype=xp.complex128)
    x.set_sample(xp.ones((4, 4), dtype=xp.complex128))
    with g.observe():
        meas = AmplitudeMeasurement(var=1e-2) @ x
    g.compile()

    rng = get_rng(seed=42)
    meas._generate_sample(rng)
    assert meas.get_sample() is not None
    meas.update_observed_from_sample()
    assert isinstance(meas.observed, UncertainArray)
    assert meas.observed.shape == (4, 4)
    assert np.issubdtype(meas.observed.data.dtype, np.floating)


@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_measurement_update_observed_with_mask(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(2, 2), dtype=xp.complex128)
    mask = xp.array([[True, False], [False, True]], dtype=bool)
    with g.observe():
        meas = AmplitudeMeasurement(var=1e-2, mask=mask) @ x
    g.compile()

    x.set_sample(xp.ones((2, 2), dtype=xp.complex128))
    meas._generate_sample(get_rng(seed=0))
    meas.update_observed_from_sample()

    prec = meas.observed.precision(raw=True)
    assert xp.allclose(prec[mask], 1.0 / 1e-2)
    assert xp.all(prec[~mask] == 0.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_measurement_set_observed_dtype_and_cast(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(2, 2), dtype=xp.complex128)
    with g.observe():
        meas = AmplitudeMeasurement(var=1e-3) @ x
    g.compile()
    data = xp.ones((2, 2), dtype=xp.float64)
    meas.set_observed(data)
    assert np.issubdtype(meas.expected_observed_dtype, np.floating)


@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_measurement_graph_to_backend_syncs_dtype_and_arrays(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(2, 2), dtype=xp.complex128)
    mask = xp.array([[True, False], [False, True]], dtype=bool)
    with g.observe():
        meas = AmplitudeMeasurement(var=1e-3, mask=mask) @ x
    g.compile()

    x.set_sample(xp.ones((2, 2), dtype=xp.complex128))
    meas._generate_sample(get_rng(seed=1))
    meas.update_observed_from_sample()

    new_backend = cp if xp is np else np
    backend.set_backend(new_backend)
    g.to_backend()

    assert isinstance(meas.observed.data, new_backend.ndarray)
    assert isinstance(meas.mask, new_backend.ndarray)
    assert isinstance(meas.input_dtype, type(new_backend.dtype(meas.input_dtype)))
    assert isinstance(meas.expected_observed_dtype, type(new_backend.dtype(meas.expected_observed_dtype)))
