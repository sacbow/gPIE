import pytest
from gpie.core.types import get_real_dtype
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
