from enum import Enum

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

try:
    from numpy.typing import NDArray
    import numpy as _np
    ArrayLike = NDArray[_np.generic]
except ImportError:
    ArrayLike = "Any"  # fallback for environments without numpy.typing

# Optional: more specific aliases
FloatArray = ArrayLike
ComplexArray = ArrayLike
