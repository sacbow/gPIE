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
    - "cupy": GPU-accelerated FFT using CuPy (if backend is set to CuPy).
    - "fftw": High-performance CPU FFT using pyFFTW (requires NumPy backend).
- Internal caching of FFTW plans and aligned buffers for efficient reuse.
- Safe copying by default to avoid side-effects, with optional in-place operation.

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

# ---- Abstract Interface ----

class FFTBackend:
    def fft2_centered(self, x: Any) -> Any:
        raise NotImplementedError

    def ifft2_centered(self, x: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


# ---- NumPy/Cupy Backend ----

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

    This class provides centered 2D FFT and IFFT operations using FFTW's
    high-performance planning and multithreading features.

    Overview
    --------
    FFTW is a "planning" FFT library. It benchmarks many different
    FFT strategies to determine the fastest method for a given array
    shape and data type. These plans are cached and reused for future
    computations with the same shape and dtype.

    Features
    --------
    - Threaded execution via `threads` parameter.
    - Plan caching keyed on (shape, dtype, direction).
    - Uses aligned memory buffers via `pyfftw.empty_aligned`.
    - Centered FFTs via ifftshift → fft2 → fftshift.

    Parameters
    ----------
    threads : int
        Number of threads to use for FFTW computations.
    planner_effort : str
        Planning strategy (e.g. "FFTW_MEASURE", "FFTW_PATIENT", etc.).
        See FFTW documentation for details.

    Notes
    -----
    - This backend requires NumPy as the numerical backend.
    - Input arrays are copied into preallocated aligned buffers.
      The plan object itself retains references to these buffers.
    - To avoid unintended mutation, results are copied by default
      unless `copy=False` is passed.

    Raises
    ------
    RuntimeError
        If FFTW is used when the global numerical backend is not NumPy.

    Examples
    --------
    >>> set_fft_backend("fftw", threads=4)
    >>> fft = get_fft_backend()
    >>> y = fft.fft2_centered(x)
    >>> x_rec = fft.ifft2_centered(y)
    """

    def __init__(self, threads: int = 1, planner_effort: str = "FFTW_MEASURE"):
        self.threads = threads
        self.planner_effort = planner_effort
        self._plans: Dict[Tuple[Tuple[int, ...], Any, str], Tuple[pyfftw.FFTW, np().ndarray, np().ndarray]] = {}

    def _get_plan(self, shape: Tuple[int, ...], dtype: Any, direction: str):
        key = (shape, dtype, direction)
        if key not in self._plans:
            # Create aligned input/output arrays
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

    def fft2_centered(self, x: ArrayLike, copy: bool = True) -> ArrayLike:
        x = np().asarray(x)
        plan, a, b = self._get_plan(x.shape, x.dtype, "fft")

        a[:] = np().fft.ifftshift(x, axes=(-2, -1))
        plan()
        result = np().fft.fftshift(b, axes=(-2, -1))
        return result.copy() if copy else result

    def ifft2_centered(self, x: ArrayLike, copy: bool = True) -> ArrayLike:
        x = np().asarray(x)
        plan, a, b = self._get_plan(x.shape, x.dtype, "ifft")

        a[:] = np().fft.ifftshift(x, axes=(-2, -1))
        plan()
        result = np().fft.fftshift(b, axes=(-2, -1))
        return result.copy() if copy else result



# ---- Global Backend Management ----

_current_fft_backend: FFTBackend = DefaultFFTBackend()

def set_fft_backend(name: str, **kwargs):
    global _current_fft_backend
    xp_name = np().__name__

    if name in ("numpy", "cupy"):
        _current_fft_backend = DefaultFFTBackend()

    elif name == "fftw":
        if xp_name != "numpy":
            raise RuntimeError("FFTW backend requires NumPy as numerical backend.")
        _current_fft_backend = FFTWBackend(**kwargs)

    else:
        raise ValueError(f"Unknown FFT backend: {name}")


def get_fft_backend() -> FFTBackend:
    return _current_fft_backend
