"""
FFT backend abstraction for gPIE.

This module provides a backend-agnostic interface for performing centered 2D FFTs 
and inverse FFTs in a way that is compatible with NumPy, CuPy, and optionally 
pyFFTW (if installed).

Key Features
------------
- Unified interface for 2D centered FFT and IFFT.
- Pluggable backends via `set_fft_backend(name)`, supporting:
    - "numpy": Default NumPy FFT backend.
    - "cupy": GPU-accelerated FFT with cuFFT plan caching.
    - "fftw": High-performance CPU FFT using pyFFTW (requires NumPy backend).
- Internal caching of FFTW/CuPy plans for efficient reuse.
- Safe copying by default to avoid side-effects.

Typical Usage
-------------
>>> from gpie.core.fft import set_fft_backend, get_fft_backend
>>> set_fft_backend("fftw", threads=4)
>>> fft = get_fft_backend()
>>> y = fft.fft2_centered(x)
>>> x_rec = fft.ifft2_centered(y)

Note
----
- The centered FFT applies `ifftshift → fft2 → fftshift` over the last two axes.
- This is commonly used in image reconstruction and inverse problems where 
  frequency-domain symmetry is assumed.

See Also
--------
- gpie.core.backend: Global numerical backend management (NumPy or CuPy).
- gpie.core.linalg_utils: FFT utility functions that delegate to this backend.
"""

from typing import Any, Tuple, Dict
from .types import ArrayLike
from ..core.backend import np, get_backend
import warnings

# Attempt to import pyfftw (optional dependency)
try:
    import pyfftw
except ImportError:
    pyfftw = None

# Attempt to import CuPy (optional dependency)
try:
    import cupy as cp
    import cupyx.scipy.fftpack as cufft
except ImportError:
    cp = None
    cufft = None


# ---- Abstract Interface ----

class FFTBackend:
    def fft2_centered(self, x: Any) -> Any:
        raise NotImplementedError

    def ifft2_centered(self, x: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


# ---- NumPy Backend ----

class DefaultFFTBackend(FFTBackend):
    def fft2_centered(self, x):
        return np().fft.fftshift(
            np().fft.fft2(
                np().fft.ifftshift(x, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho"
            ),
            axes=(-2, -1)
        )

    def ifft2_centered(self, x):
        return np().fft.fftshift(
            np().fft.ifft2(
                np().fft.ifftshift(x, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho"
            ),
            axes=(-2, -1)
        )


# ---- FFTW Backend ----

class FFTWBackend(FFTBackend):
    """
    FFT backend using pyFFTW (a Python wrapper for FFTW).
    Provides centered 2D FFT and IFFT operations with plan caching.
    """

    def __init__(self, threads: int = 1, planner_effort: str = "FFTW_MEASURE"):
        self.threads = threads
        self.planner_effort = planner_effort
        self._plans: Dict[Tuple[Tuple[int, ...], Any, str], Tuple[pyfftw.FFTW, Any, Any]] = {}

    def _get_plan(self, shape: Tuple[int, ...], dtype: Any, direction: str):
        key = (shape, dtype, direction)
        if key not in self._plans:
            a = pyfftw.empty_aligned(shape, dtype=dtype)
            b = pyfftw.empty_aligned(shape, dtype=dtype)
            fftw_dir = "FFTW_FORWARD" if direction == "fft" else "FFTW_BACKWARD"
            plan = pyfftw.FFTW(
                a, b,
                axes=(-2, -1),
                direction=fftw_dir,
                threads=self.threads,
                flags=(self.planner_effort,)
            )
            self._plans[key] = (plan, a, b)
        return self._plans[key]

    def fft2_centered(self, x: ArrayLike) -> ArrayLike:
        x = np().asarray(x)
        plan, a, b = self._get_plan(x.shape, x.dtype, "fft")
        a[:] = np().fft.ifftshift(x, axes=(-2, -1))
        plan()
        result = np().fft.fftshift(b, axes=(-2, -1))
        norm = np().sqrt(np().prod(x.shape[-2:]))
        return result / norm

    def ifft2_centered(self, x: ArrayLike) -> ArrayLike:
        x = np().asarray(x)
        plan, a, b = self._get_plan(x.shape, x.dtype, "ifft")
        a[:] = np().fft.ifftshift(x, axes=(-2, -1))
        plan()
        result = np().fft.fftshift(b, axes=(-2, -1))
        norm = np().sqrt(np().prod(x.shape[-2:]))
        return result * norm


# ---- CuPy Backend ----

class CuPyFFTBackend(FFTBackend):
    """
    FFT backend using CuPy with cuFFT plan caching.
    Provides centered 2D FFT and IFFT operations on the GPU.
    """

    def __init__(self):
        if cp is None or cufft is None:
            raise RuntimeError("CuPy is not available.")
        self._plans: Dict[Tuple[Tuple[int, ...], Any, str], Any] = {}

    def _get_plan(self, x, direction: str):
        key = (x.shape, x.dtype, direction)
        if key not in self._plans:
            axes = (-2, -1)
            self._plans[key] = cufft.get_fft_plan(x, axes=axes, value_type="C2C")
        return self._plans[key]

    def fft2_centered(self, x: ArrayLike) -> ArrayLike:
        x = cp.asarray(x)
        plan = self._get_plan(x, "fft")
        with plan:
            y = cp.fft.fftshift(
                cp.fft.fft2(cp.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm = "ortho"),
                axes=(-2, -1)
            )
        return y

    def ifft2_centered(self, x: ArrayLike) -> ArrayLike:
        x = cp.asarray(x)
        plan = self._get_plan(x, "ifft")
        with plan:
            y = cp.fft.fftshift(
                cp.fft.ifft2(cp.fft.ifftshift(x, axes=(-2, -1)), axes=(-2, -1), norm = "ortho"),
                axes=(-2, -1)
            )
        return y 


# ---- Global Backend Management ----

_current_fft_backend: FFTBackend = DefaultFFTBackend()

def set_fft_backend(name: str, **kwargs):
    global _current_fft_backend
    xp_name = np().__name__

    if name == "numpy":
        _current_fft_backend = DefaultFFTBackend()

    elif name == "cupy":
        if cp is None:
            raise RuntimeError("CuPy backend selected but CuPy is not installed.")
        _current_fft_backend = CuPyFFTBackend()

    elif name == "fftw":
        if xp_name != "numpy":
            raise RuntimeError("FFTW backend requires NumPy as numerical backend.")
        if pyfftw is None:
            raise RuntimeError("pyFFTW is not installed.")
        _current_fft_backend = FFTWBackend(**kwargs)

    else:
        raise ValueError(f"Unknown FFT backend: {name}")


def get_fft_backend() -> FFTBackend:
    return _current_fft_backend
