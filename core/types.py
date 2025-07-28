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
