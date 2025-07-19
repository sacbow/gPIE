from typing import Optional
import numpy as np
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA
from core.types import PrecisionMode, BinaryPropagatorPrecisionMode as BPM


class BinaryPropagator(Propagator):
    def __init__(self, precision_mode: Optional[BPM] = None):
        super().__init__(input_names=("a", "b"), dtype=None, precision_mode=precision_mode)

    def __matmul__(self, inputs: tuple[Wave, Wave]) -> Wave:
        if not (isinstance(inputs, tuple) and len(inputs) == 2):
            raise ValueError("BinaryPropagator requires a tuple of two Wave objects.")

        a, b = inputs
        self.add_input("a", a)
        self.add_input("b", b)

        if np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
            output_dtype = np.result_type(a.dtype, b.dtype)
        else:
            output_dtype = np.complex128

        if a.shape != b.shape:
            raise ValueError(f"Input shapes must match: got {a.shape} and {b.shape}")

        output = Wave(a.shape, dtype=output_dtype)
        self.connect_output(output)
        return output

    def _set_precision_mode(self, mode: BPM) -> None:
        allowed = {
            BPM.SCALAR,
            BPM.ARRAY,
            BPM.SCALAR_AND_ARRAY_TO_ARRAY,
            BPM.ARRAY_AND_SCALAR_TO_ARRAY,
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

        if a_mode is None and b_mode is None:
            return

        if a_mode is not None and b_mode is not None:
            if a_mode == b_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.SCALAR)
            elif a_mode == b_mode == PrecisionMode.ARRAY:
                self._set_precision_mode(BPM.ARRAY)
            elif a_mode == PrecisionMode.SCALAR and b_mode == PrecisionMode.ARRAY:
                self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
            elif a_mode == PrecisionMode.ARRAY and b_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
            else:
                raise ValueError(f"Invalid combination of precision modes: {a_mode}, {b_mode}")
        else:
            if a_mode == PrecisionMode.ARRAY or b_mode == PrecisionMode.ARRAY:
                self._set_precision_mode(BPM.ARRAY)
            elif a_mode == PrecisionMode.SCALAR or b_mode == PrecisionMode.SCALAR:
                return

    def set_precision_mode_backward(self):
        z_mode = self.output.precision_mode_enum
        a_mode = self.inputs["a"].precision_mode_enum
        b_mode = self.inputs["b"].precision_mode_enum

        if z_mode is None:
            return

        if z_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.SCALAR)
            return

        if z_mode == PrecisionMode.ARRAY:
            if a_mode is None and b_mode is None:
                self._set_precision_mode(BPM.ARRAY)
            elif a_mode is None and b_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
            elif a_mode == PrecisionMode.SCALAR and b_mode is None:
                self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
            elif a_mode == PrecisionMode.SCALAR and b_mode == PrecisionMode.SCALAR:
                raise ValueError(
                    "Inconsistent state: output is array but both inputs are scalar."
                )
            elif a_mode == PrecisionMode.ARRAY or b_mode == PrecisionMode.ARRAY:
                return

        raise ValueError(
            f"Unhandled combination in set_precision_mode_backward(): "
            f"a={a_mode}, b={b_mode}, output={z_mode}"
        )

    def get_output_precision_mode(self) -> Optional[str]:
        mode = self.precision_mode_enum
        if mode is None:
            return None
        return "scalar" if mode == BPM.SCALAR else "array"

    def get_input_precision_mode(self, wave: Wave) -> Optional[str]:
        mode = self.precision_mode_enum
        if mode is None:
            return None

        if mode == BPM.SCALAR:
            return "scalar"
        if mode == BPM.ARRAY:
            return "array"
        if mode == BPM.SCALAR_AND_ARRAY_TO_ARRAY:
            return "scalar" if wave is self.inputs["a"] else "array"
        if mode == BPM.ARRAY_AND_SCALAR_TO_ARRAY:
            return "array" if wave is self.inputs["a"] else "scalar"

        raise ValueError(f"Wave {wave} not recognized.")

    def forward(self):
        a_msg = self.input_messages[self.inputs["a"]]
        b_msg = self.input_messages[self.inputs["b"]]
        if a_msg is None or b_msg is None:
            raise RuntimeError("Missing input messages for BinaryPropagator.")
        output_msg = self._compute_forward({"a": a_msg, "b": b_msg})
        self.output.receive_message(self, output_msg)

    def backward(self):
        if self.output_message is None:
            raise RuntimeError("Missing output message for backward pass.")

        for exclude in ("a", "b"):
            wave = self.inputs[exclude]
            msg = self._compute_backward(self.output_message, exclude=exclude)
            wave.receive_message(self, msg)

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        raise NotImplementedError()

    def _compute_backward(self, output: UA, exclude: str) -> UA:
        raise NotImplementedError()
