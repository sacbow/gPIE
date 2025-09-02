"""
Backend abstraction for numerical computation.

This module defines a swappable backend interface for NumPy-like libraries
such as NumPy, CuPy, JAX, or PyTorch. All internal math in `core/` should use
`from .backend import np` to remain backend-agnostic.
"""

import numpy as _np

_backend = _np  # Default backend: NumPy


def set_backend(lib):
    """
    Set the global backend to a NumPy-compatible library (e.g., jax.numpy, cupy).

    Args:
        lib: Module object such as numpy, jax.numpy, or cupy.
    """
    global _backend
    _backend = lib


def get_backend():
    """
    Get the current backend module.

    Returns:
        The active backend module (default: numpy).
    """
    return _backend


# Aliases for convenience
np = get_backend  # use: `np().sum(...)`
