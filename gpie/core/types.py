from enum import Enum
from typing import Union
from numpy.typing import NDArray
from .backend import np

# === Precision Modes ===

class PrecisionMode(Enum):
    """Precision mode used in Wave and simple Factor nodes."""
    SCALAR = "scalar"
    ARRAY = "array"

    def __str__(self) -> str:
        return self.value


class UnaryPropagatorPrecisionMode(Enum):
    """Precision mode used in propagators with 1 input and 1 output."""
    SCALAR = "scalar"
    ARRAY = "array"
    SCALAR_TO_ARRAY = "scalar to array"
    ARRAY_TO_SCALAR = "array to scalar"

    def __str__(self) -> str:
        return self.value


class BinaryPropagatorPrecisionMode(Enum):
    """Precision mode used in propagators with 2 inputs (e.g., add, multiply)."""
    SCALAR = "scalar"
    ARRAY = "array"
    SCALAR_AND_ARRAY_TO_ARRAY = "scalar/array to array"
    ARRAY_AND_SCALAR_TO_ARRAY = "array/scalar to array"

    def __str__(self) -> str:
        return self.value


# === Backend-agnostic Array Type Hints ===

# dtype-agnostic array
ArrayLike = Union[
    NDArray["float64"], 
    NDArray["complex128"]
]

# Scalar or array-valued precision
Precision = Union[
    float,
    NDArray["float64"]
]

# complex128/64 to float 64/32
def get_real_dtype(dtype: np().dtype) -> np().dtype:
    if np().issubdtype(dtype, np().complexfloating):
        return np().float32 if dtype == np().complex64 else np().float64
    elif np().issubdtype(dtype, np().floating):
        return dtype
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

# float 64/32 to complex 128/64
def get_complex_dtype(dtype: np().dtype) -> np().dtype:
    if np().issubdtype(dtype, np().complexfloating):
        return dtype
    elif np().issubdtype(dtype, np().floating):
        return np().complex64 if dtype == np().float32 else np().complex128
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")

def get_lower_precision_dtype(dtype_a: np().dtype, dtype_b: np().dtype) -> np().dtype:
    """
    Return the lower precision dtype between two dtypes.
    Handles float and complex dtypes, preferring:
        float32 < float64
        complex64 < complex128
    Mixing float and complex will result in complex of lower precision.
    """
    # Normalize to np().dtype
    dtype_a = np().dtype(dtype_a)
    dtype_b = np().dtype(dtype_b)

    # Both complex
    if np().issubdtype(dtype_a, np().complexfloating) and np().issubdtype(dtype_b, np().complexfloating):
        return np().complex64 if (dtype_a == np().complex64 or dtype_b == np().complex64) else np().complex128

    # Both float
    if np().issubdtype(dtype_a, np().floating) and np().issubdtype(dtype_b, np().floating):
        return np().float32 if (dtype_a == np().float32 or dtype_b == np().float32) else np().float64

    # Float vs Complex: pick complex of lower precision
    if np().issubdtype(dtype_a, np().floating) and np().issubdtype(dtype_b, np().complexfloating):
        return np().complex64 if dtype_b == np().complex64 or dtype_a == np().float32 else np().complex128
    if np().issubdtype(dtype_b, np().floating) and np().issubdtype(dtype_a, np().complexfloating):
        return np().complex64 if dtype_a == np().complex64 or dtype_b == np().float32 else np().complex128

    # Integers or unexpected types fallback
    return np().result_type(dtype_a, dtype_b)
