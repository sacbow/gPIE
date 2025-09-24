"""
Backend abstraction for numerical computation.

This module defines a swappable backend interface for NumPy-like libraries
such as NumPy, CuPy, JAX, or PyTorch. All internal math in `core/` should use
`from .backend import np` to remain backend-agnostic.
"""

import numpy as _np
from typing import Any, Optional

_backend = _np  # Default backend: NumPy


def set_backend(lib):
    """
    Set the global backend to a NumPy-compatible library (e.g., jax.numpy, cupy).

    Args:
        lib: Module object such as numpy, jax.numpy, or cupy.
    """
    from .fft import DefaultFFTBackend
    global _backend, _current_fft_backend
    _backend = lib
    _current_fft_backend = DefaultFFTBackend()


def get_backend():
    """
    Get the current backend module.

    Returns:
        The active backend module (default: numpy).
    """
    return _backend

def move_array_to_current_backend(array: Any, dtype: Optional[_np.dtype] = None) -> Any:
    """
    Ensure array is on the current backend (NumPy or CuPy), with optional dtype conversion.

    Args:
        array (Any): Input array from potentially another backend.
        dtype (np().dtype, optional): If given, cast to this dtype after transfer.

    Returns:
        backend ndarray: Array on current backend, dtype adjusted if needed.
    """
    try:
        import cupy as cp
        # If moving from CuPy to NumPy
        if isinstance(array, cp.ndarray) and np().__name__ == "numpy":
            array = array.get()
    except ImportError:
        pass

    arr = np().asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr



# Aliases for convenience
np = get_backend  # use: `np().sum(...)`
