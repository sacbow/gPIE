import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
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
def test_gaussian_measurement_init_and_dtype_inference(xp):
    backend.set_backend(xp)
    obs = xp.ones((4, 4), dtype=xp.float32)
    meas = GaussianMeasurement(observed_array=obs, var=0.5)
    assert meas.observed is not None
    assert meas.observed.data.dtype == xp.float32
    assert meas.expected_observed_dtype == xp.float32


@pytest.mark.parametrize("xp", backend_libs)
def test_gaussian_measurement_incompatible_dtype_raises(xp):
    backend.set_backend(xp)
    obs = xp.ones((2, 2), dtype=xp.float32)
    # 明示的にcomplex128を指定しつつ実数観測を渡す → TypeError
    with pytest.raises(TypeError):
        GaussianMeasurement(observed_array=obs, var=1.0, dtype=xp.complex128)


@pytest.mark.parametrize("xp", backend_libs)
def test_gaussian_measurement_mask_and_precision(xp):
    backend.set_backend(xp)
    obs = xp.ones((3, 3), dtype=xp.float64)
    mask = xp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    meas = GaussianMeasurement(observed_array=obs, var=2.0, mask=mask, precision_mode="array")
    assert meas.mask is not None
    assert meas.observed.precision_mode == PrecisionMode.ARRAY
    prec = meas.observed.precision(raw=True)
    assert xp.allclose(prec[mask], 1.0 / 2.0)
    assert xp.all(prec[~mask] == 0.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_gaussian_measurement_generate_and_update_observed(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(4, 4), dtype=xp.float32)
    x.set_sample(xp.zeros((4, 4), dtype=xp.float32))
    meas = GaussianMeasurement(var=0.1, dtype=xp.float32) @ x
    g.compile()

    rng = get_rng(seed=42)
    meas._generate_sample(rng)
    assert meas.get_sample() is not None
    meas.update_observed_from_sample()
    assert isinstance(meas.observed, UncertainArray)
    assert meas.observed.shape == (4, 4)


@pytest.mark.parametrize("xp", backend_libs)
def test_gaussian_measurement_set_observed_dtype_infer_and_cast(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(2, 2), dtype=xp.float32)
    with g.observe():
        meas = GaussianMeasurement(var=0.1) @ x
    g.compile()

    data = xp.ones((2, 2), dtype=xp.float64)
    meas.set_observed(data)
    assert meas.expected_observed_dtype == xp.float64

    data2 = xp.ones((2, 2), dtype=xp.float32)
    meas.set_observed(data2)
    assert meas.observed.data.dtype == xp.float64 


@pytest.mark.parametrize("xp", backend_libs)
def test_gaussian_measurement_set_observed_guard_input_dtype(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(2, 2), dtype=xp.float32)
    with g.observe():
        meas = GaussianMeasurement(var=0.1) @ x
    g.compile()

    bad_data = xp.ones((2, 2), dtype=xp.complex64)  
    with pytest.raises(TypeError):
        meas.set_observed(bad_data)


@pytest.mark.parametrize("xp", backend_libs)
def test_gaussian_measurement_graph_to_backend_syncs_dtype_and_arrays(xp):
    backend.set_backend(xp)
    g = Graph()
    x = Wave(shape=(4, 4), dtype=xp.float32)
    x.set_sample(xp.zeros((4, 4), dtype=xp.float32))

    mask = xp.ones((4, 4), dtype=bool)
    with g.observe():
        meas = GaussianMeasurement(var=0.1, dtype=xp.float32, mask=mask) @ x
    g.compile()
    rng = get_rng(seed=42)
    meas._generate_sample(rng)
    meas.update_observed_from_sample()

    new_backend = cp if xp is np else np
    backend.set_backend(new_backend)
    g.to_backend()

    assert isinstance(meas.observed.data, new_backend.ndarray)
    assert isinstance(meas.mask, new_backend.ndarray)
    assert isinstance(meas.input_dtype, type(new_backend.dtype(meas.input_dtype)))
    assert isinstance(meas.expected_observed_dtype, type(new_backend.dtype(meas.expected_observed_dtype)))
