import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.core.linalg_utils import (
    reduce_precision_to_scalar,
    complex_normal_random_array,
    random_normal_array,
    sparse_complex_array,
    random_unitary_matrix,
    random_binary_mask,
    random_phase_mask,
    circular_aperture,
    square_aperture,
    fft2_centered,
    ifft2_centered,
    masked_random_array,
    angular_spectrum_phase_mask,
)
import warnings

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_reduce_precision_to_scalar_valid_and_invalid(xp):
    backend.set_backend(xp)
    arr = xp.array([1.0, 2.0, 4.0])
    scalar = reduce_precision_to_scalar(arr)
    expected = 1.0 / xp.mean(1.0 / arr)
    assert np.isclose(scalar, expected)

    with pytest.raises(ValueError):
        reduce_precision_to_scalar(xp.array([1.0, -1.0]))


@pytest.mark.parametrize("xp", backend_libs)
def test_complex_normal_random_array_warns_and_shape(xp):
    backend.set_backend(xp)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        arr = complex_normal_random_array((4, 4), dtype=xp.complex128)
        assert arr.shape == (4, 4)
        assert xp.iscomplexobj(arr)
        assert any("deprecated" in str(wi.message).lower() for wi in w)


@pytest.mark.parametrize("xp", backend_libs)
def test_random_normal_array_real_and_complex_and_invalid(xp):
    backend.set_backend(xp)
    rng = np.random.default_rng(0)
    c = random_normal_array((3,), dtype=xp.complex128, rng=rng)
    assert xp.iscomplexobj(c)

    r = random_normal_array((3,), dtype=xp.float32, rng=rng)
    assert r.dtype == xp.float32

    with pytest.raises(ValueError):
        random_normal_array((2,), dtype=xp.int32)


@pytest.mark.parametrize("xp", backend_libs)
def test_sparse_complex_array(xp):
    backend.set_backend(xp)
    arr = sparse_complex_array((4,), sparsity=0.5)
    assert arr.shape == (4,)
    assert xp.iscomplexobj(arr)
    zeros = xp.sum(arr == 0)
    assert 0 < zeros < arr.size


@pytest.mark.parametrize("xp", backend_libs)
def test_random_unitary_matrix(xp):
    backend.set_backend(xp)
    U = random_unitary_matrix(4, dtype=xp.complex128)
    assert U.shape == (4, 4)
    I = U.conj().T @ U
    assert xp.allclose(I, xp.eye(4), atol=1e-12)


@pytest.mark.parametrize("xp", backend_libs)
def test_random_binary_mask(xp):
    backend.set_backend(xp)
    mask = random_binary_mask((8,), subsampling_rate=0.25)
    assert mask.shape == (8,)
    assert mask.dtype == bool
    assert 0 < mask.sum() < 8

    mask2 = random_binary_mask(4)
    assert mask2.shape == (4,)


@pytest.mark.parametrize("xp", backend_libs)
def test_random_phase_mask(xp):
    backend.set_backend(xp)
    mask = random_phase_mask((4, 4))
    assert mask.shape == (4, 4)
    assert xp.allclose(xp.abs(mask), 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_circular_aperture_valid_and_invalid(xp):
    backend.set_backend(xp)
    mask = circular_aperture((10, 10), radius=0.3)
    assert mask.shape == (10, 10)
    assert mask.dtype == bool

    with pytest.raises(ValueError):
        circular_aperture((10, 10), radius=0.6)

    with pytest.raises(ValueError):
        circular_aperture((10, 10), radius=0.3, center=(2.0, 2.0))


@pytest.mark.parametrize("xp", backend_libs)
def test_square_aperture_valid_and_invalid(xp):
    backend.set_backend(xp)
    mask = square_aperture((10, 10), radius=0.3)
    assert mask.shape == (10, 10)
    assert mask.dtype == bool

    with pytest.raises(ValueError):
        square_aperture((10, 10), radius=0.6)

    with pytest.raises(ValueError):
        square_aperture((5, 5), radius=0.49, center=(0.4, 0.4))


@pytest.mark.parametrize("xp", backend_libs)
def test_fft2_and_ifft2_centered(xp):
    backend.set_backend(xp)
    x = xp.random.rand(8, 8)
    X = fft2_centered(x)
    x_rec = ifft2_centered(X)
    assert xp.allclose(x, x_rec, atol=1e-12)


@pytest.mark.parametrize("xp", backend_libs)
def test_masked_random_array(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])
    arr = masked_random_array(support, dtype=xp.complex128)
    assert arr.shape == support.shape
    assert arr[0, 1] == 0


@pytest.mark.parametrize("xp", backend_libs)
def test_angular_spectrum_phase_mask(xp):
    backend.set_backend(xp)
    mask = angular_spectrum_phase_mask((8, 8), wavelength=500e-9, distance=0.01, dx=1e-6)
    assert mask.shape == (8, 8)
    assert xp.iscomplexobj(mask)
    assert xp.allclose(xp.abs(mask), 1.0, atol=1e-12)
