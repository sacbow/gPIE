from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.backend import np
from ...core.types import PrecisionMode, BinaryPropagatorPrecisionMode as BPM, get_lower_precision_dtype


class BinaryPropagator(Propagator):
    """
    Abstract base class for two-input propagators in the computational graph.

    This class serves as a parent for operations like addition, multiplication,
    or other binary transformations with two Wave inputs ("a" and "b") and a single output.

    It handles:
        - Input/output wiring with shape and dtype checks
        - Precision mode inference and compatibility logic
        - Message routing and EP-style forward/backward logic

    Supported precision modes:
        - SCALAR: All variables have scalar precision
        - ARRAY: All variables use elementwise precision
        - SCALAR_AND_ARRAY_TO_ARRAY: Mixed mode (scalar + array → array)
        - ARRAY_AND_SCALAR_TO_ARRAY: Mixed mode (array + scalar → array)

    Subclasses must implement:
        - `_compute_forward(inputs: dict[str, UA]) -> UA`
        - `_compute_backward(output: UA, exclude: str) -> UA`

    Usage:
        z = AddPropagator() @ (x, y)
    """

    def __init__(self, precision_mode: Optional[BPM] = None):
        super().__init__(input_names=("a", "b"), dtype=None, precision_mode=precision_mode)
        self._init_rng = None
    
    def set_init_rng(self, rng):
        self._init_rng = rng

    def __matmul__(self, inputs: tuple[Wave, Wave]) -> Wave:
        """
        Wire two input Wave nodes into the binary propagator.

        Performs:
            - Shape compatibility check (batch_size and event_shape)
            - dtype resolution (real → real, mixed → complex)
            - Connection of inputs and output
            - Graph generation index setup

        Args:
            inputs (tuple[Wave, Wave]): Two wave variables to combine.

        Returns:
            Wave: The output wave resulting from the binary operation.

        Raises:
            ValueError: If inputs are invalid or incompatible.
        """

        if not (isinstance(inputs, tuple) and len(inputs) == 2):
            raise ValueError("BinaryPropagator requires a tuple of two Wave objects.")

        a, b = inputs
        self.add_input("a", a)
        self.add_input("b", b)

        # Resolve output dtype (e.g., float + complex → complex)
        self.dtype = get_lower_precision_dtype(a.dtype, b.dtype)

        # Check compatibility: batch size and event shape
        if a.batch_size != b.batch_size:
            raise ValueError(f"Batch size mismatch: {a.batch_size} vs {b.batch_size}")
        if a.event_shape != b.event_shape:
            raise ValueError(f"Event shape mismatch: {a.event_shape} vs {b.event_shape}")
        self.batch_size = a.batch_size
        # Create output wave
        output = Wave(
            event_shape=a.event_shape,
            batch_size=a.batch_size,
            dtype=self.dtype
        )
        self.connect_output(output)
        return output


    def _set_precision_mode(self, mode: BPM) -> None:
        allowed = {
            BPM.SCALAR,
            BPM.ARRAY,
            BPM.SCALAR_AND_ARRAY_TO_ARRAY,
            BPM.ARRAY_AND_SCALAR_TO_ARRAY,
            BPM.ARRAY_AND_ARRAY_TO_SCALAR
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for BinaryPropagator: '{mode}'")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict in BinaryPropagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self):
        a_mode = self.inputs["a"].precision_mode_enum
        b_mode = self.inputs["b"].precision_mode_enum

        if a_mode == PrecisionMode.SCALAR and b_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.SCALAR)
        if a_mode == PrecisionMode.SCALAR and b_mode == PrecisionMode.ARRAY:
            self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
        if a_mode == PrecisionMode.ARRAY and b_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
        else:
            return

    def set_precision_mode_backward(self):
        z_mode = self.output.precision_mode_enum
        a_mode = self.inputs["a"].precision_mode_enum
        b_mode = self.inputs["b"].precision_mode_enum

        if z_mode == PrecisionMode.ARRAY:
            if a_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
            elif b_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
            else:
                self._set_precision_mode(BPM.ARRAY)
        
        else:
            if a_mode == PrecisionMode.ARRAY or b_mode == PrecisionMode.ARRAY:
                self._set_precision_mode(BPM.ARRAY_AND_ARRAY_TO_SCALAR)
            else:
                self._set_precision_mode(BPM.SCALAR)


    def get_output_precision_mode(self) -> Optional[str]:
        mode = self.precision_mode_enum
        if mode is None:
            return None
        if mode == BPM.SCALAR or mode == BPM.ARRAY_AND_ARRAY_TO_SCALAR:
            return "scalar"
        else:
            return "array"

    def get_input_precision_mode(self, wave: Wave) -> Optional[str]:
        mode = self.precision_mode_enum
        if mode is None:
            return None

        if mode == BPM.SCALAR:
            return "scalar"
        if mode == BPM.ARRAY or mode == BPM.ARRAY_AND_ARRAY_TO_SCALAR:
            return "array"
        if mode == BPM.SCALAR_AND_ARRAY_TO_ARRAY:
            return "scalar" if wave is self.inputs["a"] else "array"
        if mode == BPM.ARRAY_AND_SCALAR_TO_ARRAY:
            return "array" if wave is self.inputs["a"] else "scalar"

        raise ValueError(f"Wave {wave} not recognized.")
