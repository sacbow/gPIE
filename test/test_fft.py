import numpy as np
import pytest

from gpie.core.fft import (
    set_fft_backend,
    get_fft_backend,
    FFTWBackend,
    DefaultFFTBackend,
)

# Check if pyfftw is available
try:
    import pyfftw
    has_pyfftw = True
except ImportError:
    has_pyfftw = False


@pytest.mark.parametrize("backend_name", ["numpy", "fftw"] if has_pyfftw else ["numpy"])
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_fft2_ifft2_centered_identity(backend_name, shape, dtype):
    set_fft_backend(backend_name)
    fft = get_fft_backend()

    x = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    x = x.astype(dtype)

    x_rec = fft.ifft2_centered(fft.fft2_centered(x))
    assert np.allclose(x, x_rec, atol=1e-5), f"Reconstruction failed for {backend_name}, dtype={dtype}"


def test_fft_backend_switching():
    set_fft_backend("numpy")
    assert isinstance(get_fft_backend(), DefaultFFTBackend)

    if has_pyfftw:
        set_fft_backend("fftw", threads=2)
        assert isinstance(get_fft_backend(), FFTWBackend)


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        set_fft_backend("invalid_backend")


def test_fftw_requires_numpy_backend():
    if not has_pyfftw:
        pytest.skip("pyFFTW not available")

    # Simulate a non-numpy backend
    from gpie.core import backend

    class DummyBackend:
        __name__ = "not_numpy"

    original_backend = backend.get_backend()
    backend.set_backend(np)  # use NumPy to ensure FFTW works
    set_fft_backend("fftw")  # should succeed

    backend.set_backend(DummyBackend)
    with pytest.raises(RuntimeError):
        set_fft_backend("fftw")

    # Restore original backend
    backend.set_backend(original_backend)

def teardown_function():
    set_fft_backend("numpy")  # Reset to default after each test
