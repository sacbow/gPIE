import pytest
import numpy as np
import cupy as cp

from gpie.core import backend
from gpie.core.linalg_utils import (
    random_normal_array,
    random_unitary_matrix,
    reduce_precision_to_scalar,
)

@pytest.mark.parametrize("xp", [np, cp])
def test_random_normal_array_shape_and_dtype(xp):
    backend.set_backend(xp)
    shape = (16, 16)
    arr = random_normal_array(shape, dtype=xp.complex128)
    assert arr.shape == shape
    assert arr.dtype == xp.complex128
    assert isinstance(arr, xp.ndarray)

@pytest.mark.parametrize("xp", [np, cp])
def test_random_unitary_matrix_unitarity(xp):
    backend.set_backend(xp)
    n = 8
    U = random_unitary_matrix(n, dtype=xp.complex128)
    U_dagger = xp.conj(U.T)
    identity = xp.matmul(U, U_dagger)
    I = xp.eye(n, dtype=xp.complex128)
    assert xp.allclose(identity, I, atol=1e-6)

@pytest.mark.parametrize("xp", [np, cp])
def test_reduce_precision_to_scalar_correctness(xp):
    backend.set_backend(xp)
    precisions = xp.array([1.0, 2.0, 4.0])
    result = reduce_precision_to_scalar(precisions)
    expected = 1.0 / xp.mean(1.0 / precisions)
    assert xp.isclose(result, expected)
