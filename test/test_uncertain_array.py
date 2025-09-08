import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode, get_real_dtype

# Optional CuPy
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_init_scalar_and_array_precision_vectorized(xp):
    backend.set_backend(xp)
    ua_scalar = UncertainArray.zeros(event_shape=(4, 4), batch_size=3, precision=2.0, scalar_precision=True)
    ua_array = UncertainArray.zeros(event_shape=(4, 4), batch_size=3, precision=2.0, scalar_precision=False)

    assert ua_scalar.batch_size == 3
    assert ua_scalar.event_shape == (4, 4)
    assert ua_scalar.precision_mode == PrecisionMode.SCALAR
    assert np.allclose(ua_scalar.precision(), 2.0)

    assert ua_array.precision_mode == PrecisionMode.ARRAY
    assert ua_array.precision().shape == (3, 4, 4)


@pytest.mark.parametrize("xp", backend_libs)
def test_mul_and_div_vectorized(xp):
    backend.set_backend(xp)
    ua1 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=1.0)
    ua2 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=2.0)
    ua_mul = ua1 * ua2
    ua_recovered = ua_mul / ua2

    assert np.allclose(ua1.data, ua_recovered.data, atol=1e-5)


@pytest.mark.parametrize("xp", backend_libs)
def test_damp_with_extremes(xp):
    backend.set_backend(xp)
    ua1 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=1.0)
    ua2 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=10.0)

    ua_0 = ua1.damp_with(ua2, alpha=0.0)
    ua_1 = ua1.damp_with(ua2, alpha=1.0)

    assert np.allclose(ua_0.data, ua1.data)
    assert np.allclose(ua_1.data, ua2.data)


@pytest.mark.parametrize("xp", backend_libs)
def test_product_reduce_over_batch(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(4, 4), batch_size=10, precision=2.0)
    reduced = ua.product_reduce_over_batch()

    assert reduced.event_shape == (4, 4)
    assert np.allclose(reduced.precision(), np.sum(ua.precision(), axis=0), atol=1e-5)


@pytest.mark.parametrize("xp", backend_libs)
def test_as_precision_roundtrip_vectorized(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(3, 3), batch_size=4, precision=2.0, scalar_precision=True)

    assert ua.precision_mode == PrecisionMode.SCALAR
    ua_array = ua.as_array_precision()
    assert ua_array.precision_mode == PrecisionMode.ARRAY
    ua_back = ua_array.as_scalar_precision()
    assert ua_back.precision_mode == PrecisionMode.SCALAR


@pytest.mark.parametrize("xp", backend_libs)
def test_repr_contains_batch_info(xp):
    backend.set_backend(xp)
    ua = UncertainArray.zeros(event_shape=(8,), batch_size=5)
    r = repr(ua)
    assert "batch_size=5" in r
    assert "event_shape=(8,)" in r


@pytest.mark.parametrize("xp", backend_libs)
def test_shape_ndim_warns(xp):
    backend.set_backend(xp)
    ua = UncertainArray.zeros(event_shape=(4, 4))
    with pytest.warns(DeprecationWarning):
        _ = ua.shape
    with pytest.warns(DeprecationWarning):
        _ = ua.ndim


@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_roundtrip(xp):
    import numpy as np
    import cupy as cp
    if not has_cupy:
        pytest.skip("CuPy not installed")

    backend.set_backend(np)
    ua = UncertainArray.zeros(event_shape=(2, 2), dtype=np.complex64, precision=1.0)
    backend.set_backend(cp)
    ua.to_backend()
    assert isinstance(ua.data, cp.ndarray)
    assert ua.dtype == cp.complex64
