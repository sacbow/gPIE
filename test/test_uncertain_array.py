import importlib.util
import numpy as np
import pytest
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode
from gpie.core.types import get_real_dtype

def dtype_cases(lib):
    return [
        (lib.float32, lib.float32),
        (lib.float64, lib.float64),
        (lib.complex64, lib.float32),
        (lib.complex128, lib.float64),
    ]

@pytest.mark.parametrize("backend_lib", backend_libs)
def test_scalar_precision_properties(backend_lib):
    backend.set_backend(backend_lib)
    ua = UncertainArray(backend_lib.ones((4, 4)), precision=2.0)
    assert ua.shape == (4, 4)
    assert ua.ndim == 2
    assert ua.precision_mode == PrecisionMode.SCALAR
    assert ua.precision(raw=True) == 2.0
    assert np.allclose(ua.precision(), 2.0)


@pytest.mark.parametrize("backend_lib", backend_libs)
def test_array_precision_properties(backend_lib):
    backend.set_backend(backend_lib)
    precision = backend_lib.full((4, 4), 3.0)
    ua = UncertainArray(backend_lib.ones((4, 4)), precision=precision)
    assert ua.precision_mode == PrecisionMode.ARRAY
    assert np.allclose(ua.precision(), 3.0)


@pytest.mark.parametrize("backend_lib", backend_libs)
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


@pytest.mark.parametrize("backend_lib", backend_libs)
def test_mul_div_operations(backend_lib):
    backend.set_backend(backend_lib)
    ua1 = UncertainArray(backend_lib.ones((2, 2)), precision=1.0)
    ua2 = UncertainArray(backend_lib.full((2, 2), 2.0), precision=3.0)
    ua3 = ua1 * ua2
    ua4 = ua3 / ua2
    assert np.allclose(ua4.data, ua1.data)
    assert np.allclose(ua4.precision(), ua1.precision())


@pytest.mark.parametrize("backend_lib", backend_libs)
def test_damp_with_behavior(backend_lib):
    backend.set_backend(backend_lib)
    ua1 = UncertainArray(backend_lib.ones((2, 2)), precision=1.0)
    ua2 = UncertainArray(backend_lib.full((2, 2), 2.0), precision=4.0)
    ua_damped = ua1.damp_with(ua2, alpha=0.5)
    assert np.allclose(ua_damped.data, 1.5)
    assert ua_damped.precision_mode == PrecisionMode.SCALAR


@pytest.mark.parametrize("backend_lib", backend_libs)
def test_random_and_zeros_constructors(backend_lib):
    backend.set_backend(backend_lib)
    ua_r = UncertainArray.random((3, 3), dtype=backend_lib.float64, precision=2.0)
    ua_z = UncertainArray.zeros((3, 3), dtype=backend_lib.float64, precision=2.0)
    assert ua_r.shape == (3, 3)
    assert ua_z.shape == (3, 3)
    assert np.allclose(ua_z.data, 0.0)


@pytest.mark.parametrize("backend_lib", backend_libs)
def test_as_precision_conversions(backend_lib):
    backend.set_backend(backend_lib)
    prec_array = backend_lib.full((2, 2), 5.0)
    ua = UncertainArray(backend_lib.ones((2, 2)), precision=prec_array)
    ua_scalar = ua.as_scalar_precision()
    assert ua_scalar.precision_mode == PrecisionMode.SCALAR
    ua_back = ua_scalar.as_array_precision()
    assert ua_back.precision_mode == PrecisionMode.ARRAY

def test_to_backend_updates_dtype_correctly():
    from gpie.core.backend import set_backend
    import numpy as np
    import cupy as cp

    set_backend(np)
    ua = UncertainArray(np.ones((2, 2), dtype=np.complex128), precision=1.0)

    set_backend(cp)
    ua.to_backend()

    assert isinstance(ua.data, cp.ndarray)
    assert ua.dtype == cp.complex128

@pytest.mark.parametrize("backend_lib", backend_libs)
@pytest.mark.parametrize("dtype, expected_real_dtype", dtype_cases(np))
def test_get_real_dtype_matches_expectation(backend_lib, dtype, expected_real_dtype):
    backend.set_backend(backend_lib)
    assert get_real_dtype(dtype) == expected_real_dtype

@pytest.mark.parametrize("backend_lib", backend_libs)
@pytest.mark.parametrize("dtype, expected_real_dtype", dtype_cases(np))
def test_uncertain_array_dtype_and_precision_alignment(backend_lib, dtype, expected_real_dtype):
    backend.set_backend(backend_lib)
    shape = (2, 2)
    ua = UncertainArray(backend_lib.ones(shape, dtype=dtype), dtype=dtype, precision=1.0)
    
    # 型チェック
    assert ua.dtype == dtype
    assert ua.data.dtype == dtype
    
    # precision の型が対応する実数型
    raw_prec = ua.precision(raw=True)
    if ua.precision_mode.name == "SCALAR":
        assert isinstance(raw_prec, (float, int, backend_lib.ndarray))  # rank-0 array
    else:
        assert raw_prec.dtype == expected_real_dtype
        assert raw_prec.shape == shape