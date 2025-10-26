import numpy as np
import pytest
from gpie.core.backend import set_backend
from gpie.core.fft import (
    set_fft_backend,
    get_fft_backend,
    FFTWBackend,
    DefaultFFTBackend,
)

# Optional deps
try:
    import pyfftw
    has_pyfftw = True
except ImportError:
    has_pyfftw = False

try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False

set_backend(np)


@pytest.mark.parametrize(
    "backend_name",
    ["numpy"]
    + (["fftw"] if has_pyfftw else [])
    + (["cupy"] if has_cupy else []),
)
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_fft2_ifft2_centered_identity(backend_name, shape, dtype):
    set_backend(np)
    set_fft_backend(backend_name)
    fft = get_fft_backend()

    # Use appropriate backend for array creation
    if backend_name == "cupy":
        xp = cp
    else:
        xp = np

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    x_np = x_np.astype(dtype)

    x = xp.asarray(x_np)

    x_rec = fft.ifft2_centered(fft.fft2_centered(x))
    # Move back to numpy for comparison
    if backend_name == "cupy":
        x_rec = cp.asnumpy(x_rec)

    assert np.allclose(x_np, x_rec, atol=1e-5), (
        f"Reconstruction failed for {backend_name}, dtype={dtype}"
    )


def test_fft_backend_switching():
    set_fft_backend("numpy")
    assert isinstance(get_fft_backend(), DefaultFFTBackend)

    if has_pyfftw:
        set_fft_backend("fftw", threads=2)
        assert isinstance(get_fft_backend(), FFTWBackend)

    if has_cupy:
        set_fft_backend("cupy")
        fft = get_fft_backend()
        # Avoid import inside fft.py for isinstance check
        assert fft.__class__.__name__ == "CuPyFFTBackend"


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        set_fft_backend("invalid_backend")


from types import SimpleNamespace

def test_fftw_requires_numpy_backend():
    if not has_pyfftw:
        pytest.skip("pyFFTW not available")

    from gpie.core import backend

    original_backend = backend.get_backend()

    backend.set_backend(np)  
    set_fft_backend("fftw")

    dummy_module = SimpleNamespace(__name__="not_numpy")
    backend.set_backend(dummy_module)

    with pytest.raises(RuntimeError):
        set_fft_backend("fftw")

    backend.set_backend(original_backend)


def teardown_function():
    # Reset to default after each test
    set_fft_backend("numpy")
