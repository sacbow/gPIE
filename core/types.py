from enum import Enum


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
    SCALAR_OR_ARRAY_TO_ARRAY = "scalar/array to array"

    def __str__(self) -> str:
        return self.value
