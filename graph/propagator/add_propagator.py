import numpy as np
from typing import Optional
from .binary_propagator import BinaryPropagator
from core.uncertain_array import UncertainArray as UA
from core.types import BinaryPropagatorPrecisionMode as BPM


class AddPropagator(BinaryPropagator):
    """
    A propagator that models Z = A + B under Gaussian belief propagation.
    """

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        a = inputs.get("a")
        b = inputs.get("b")

        if a is None or b is None:
            raise RuntimeError("Missing input messages: a or b is None.")

        target_dtype = np.result_type(a.dtype, b.dtype)
        if a.dtype != target_dtype:
            a = a.astype(target_dtype)
        if b.dtype != target_dtype:
            b = b.astype(target_dtype)

        mu = a.data + b.data
        precision = 1.0 / (1.0 / a._precision + 1.0 / b._precision)

        return UA(mu, dtype=target_dtype, precision=precision)


    def _compute_backward(self, output: UA, exclude: str) -> UA:
        other_name = "b" if exclude == "a" else "a"
        other_wave = self.inputs.get(other_name)
        other_msg = self.input_messages.get(other_wave)

        if other_wave is None:
            raise RuntimeError(f"Input wave '{other_name}' not found.")
        if other_msg is None:
            raise RuntimeError(f"Missing message from input wave '{other_name}'.")

        mu = output.data - other_msg.data
        precision = 1.0 / (1.0 / output._precision + 1.0 / other_msg._precision)
        out_dtype = np.result_type(output.dtype, other_msg.dtype)
        msg = UA(mu, dtype=out_dtype, precision=precision)

        mode = self.precision_mode_enum

        # Handle scalar projection if mixed precision
        if mode == BPM.SCALAR_AND_ARRAY_TO_ARRAY and exclude == "a":
            a_wave = self.inputs.get("a")
            a_msg = self.input_messages.get(a_wave)
            if a_msg is None:
                raise RuntimeError("Missing input message from wave 'a'.")
            return (a_msg * msg).as_scalar_precision() / a_msg

        if mode == BPM.ARRAY_AND_SCALAR_TO_ARRAY and exclude == "b":
            b_wave = self.inputs.get("b")
            b_msg = self.input_messages.get(b_wave)
            if b_msg is None:
                raise RuntimeError("Missing input message from wave 'b'.")
            return (b_msg * msg).as_scalar_precision() / b_msg

        return msg

    def generate_sample(self, rng):
        a = self.inputs["a"].get_sample()
        b = self.inputs["b"].get_sample()
        if a is None or b is None:
            raise RuntimeError("Input sample(s) not set for AddPropagator.")
        self.output.set_sample(a + b)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode_enum.value if self._precision_mode is not None else None
        return f"Add(gen={gen}, mode={mode})"
