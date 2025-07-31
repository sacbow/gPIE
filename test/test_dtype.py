import pytest
from gpie.core.types import get_real_dtype, get_lower_precision_dtype
from gpie.core.backend import np

def test_get_real_dtype_from_complex128():
    dtype = np().complex128
    expected = np().float64
    assert get_real_dtype(dtype) == expected

def test_get_real_dtype_from_complex64():
    dtype = np().complex64
    expected = np().float32
    assert get_real_dtype(dtype) == expected

def test_get_real_dtype_from_float64():
    dtype = np().float64
    expected = np().float64
    assert get_real_dtype(dtype) == expected

def test_get_real_dtype_from_float32():
    dtype = np().float32
    expected = np().float32
    assert get_real_dtype(dtype) == expected

def test_get_real_dtype_raises_on_invalid_type():
    with pytest.raises(TypeError):
        get_real_dtype(np().int32)

def test_get_lower_precision_dtype_float():
    assert get_lower_precision_dtype(np().float32, np().float64) == np().float32
    assert get_lower_precision_dtype(np().float64, np().float64) == np().float64

def test_get_lower_precision_dtype_complex():
    assert get_lower_precision_dtype(np().complex64, np().complex128) == np().complex64
    assert get_lower_precision_dtype(np().complex128, np().complex128) == np().complex128

def test_get_lower_precision_dtype_float_complex():
    assert get_lower_precision_dtype(np().float32, np().complex128) == np().complex64
    assert get_lower_precision_dtype(np().float64, np().complex64) == np().complex64

def test_get_lower_precision_dtype_mixed():
    # int と float/complex混在は result_type にフォールバック
    assert get_lower_precision_dtype(np().int32, np().float32) == np().result_type(np().int32, np().float32)
    assert get_lower_precision_dtype(np().int32, np().complex64) == np().result_type(np().int32, np().complex64)
