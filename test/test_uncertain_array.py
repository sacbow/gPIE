import numpy as np
import pytest
import cupy as cp

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode


@pytest.mark.parametrize("backend_lib", [np, cp])
def test_scalar_precision_properties(backend_lib):
    backend.set_backend(backend_lib)
    ua = UncertainArray(backend_lib.ones((4, 4)), precision=2.0)
    assert ua.shape == (4, 4)
    assert ua.ndim == 2
    assert ua.precision_mode == PrecisionMode.SCALAR
    assert ua.precision(raw=True) == 2.0
    assert np.allclose(ua.precision(), 2.0)


@pytest.mark.parametrize("backend_lib", [np, cp])
def test_array_precision_properties(backend_lib):
    backend.set_backend(backend_lib)
    precision = backend_lib.full((4, 4), 3.0)
    ua = UncertainArray(backend_lib.ones((4, 4)), precision=precision)
    assert ua.precision_mode == PrecisionMode.ARRAY
    assert np.allclose(ua.precision(), 3.0)


@pytest.mark.parametrize("backend_lib", [np, cp])
def test_astype_real_to_complex_and_back(backend_lib):
    backend.set_backend(backend_lib)
    ua = UncertainArray(backend_lib.ones((2, 2), dtype=backend_lib.float64), dtype = backend_lib.float64, precision=4.0)
    assert ua.is_real()
    ua_c = ua.astype(backend_lib.complex128)
    assert ua_c.is_complex()
    assert np.allclose(ua_c.precision(raw=True), 2.0)
    ua_back = ua_c.astype(backend_lib.float64)
    assert ua_back.is_real()
    assert np.allclose(ua_back.precision(raw=True), 4.0)


@pytest.mark.parametrize("backend_lib", [np, cp])
def test_mul_div_operations(backend_lib):
    backend.set_backend(backend_lib)
    ua1 = UncertainArray(backend_lib.ones((2, 2)), precision=1.0)
    ua2 = UncertainArray(backend_lib.full((2, 2), 2.0), precision=3.0)
    ua3 = ua1 * ua2
    ua4 = ua3 / ua2
    assert np.allclose(ua4.data, ua1.data)
    assert np.allclose(ua4.precision(), ua1.precision())


@pytest.mark.parametrize("backend_lib", [np, cp])
def test_damp_with_behavior(backend_lib):
    backend.set_backend(backend_lib)
    ua1 = UncertainArray(backend_lib.ones((2, 2)), precision=1.0)
    ua2 = UncertainArray(backend_lib.full((2, 2), 2.0), precision=4.0)
    ua_damped = ua1.damp_with(ua2, alpha=0.5)
    assert np.allclose(ua_damped.data, 1.5)
    assert ua_damped.precision_mode == PrecisionMode.SCALAR


@pytest.mark.parametrize("backend_lib", [np, cp])
def test_random_and_zeros_constructors(backend_lib):
    backend.set_backend(backend_lib)
    ua_r = UncertainArray.random((3, 3), dtype=backend_lib.float64, precision=2.0)
    ua_z = UncertainArray.zeros((3, 3), dtype=backend_lib.float64, precision=2.0)
    assert ua_r.shape == (3, 3)
    assert ua_z.shape == (3, 3)
    assert np.allclose(ua_z.data, 0.0)


@pytest.mark.parametrize("backend_lib", [np, cp])
def test_as_precision_conversions(backend_lib):
    backend.set_backend(backend_lib)
    prec_array = backend_lib.full((2, 2), 5.0)
    ua = UncertainArray(backend_lib.ones((2, 2)), precision=prec_array)
    ua_scalar = ua.as_scalar_precision()
    assert ua_scalar.precision_mode == PrecisionMode.SCALAR
    ua_back = ua_scalar.as_array_precision()
    assert ua_back.precision_mode == PrecisionMode.ARRAY
