from typing import Optional
from .binary_propagator import BinaryPropagator
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import BinaryPropagatorPrecisionMode as BPM, get_lower_precision_dtype, get_complex_dtype


class AddPropagator(BinaryPropagator):
    """
    A propagator representing the sum of two latent variables: Z = A + B.

    This propagator fuses Gaussian beliefs of two inputs (`a`, `b`) and produces
    a new belief for the output `z`. It supports both scalar and array precision modes,
    including asymmetric combinations (e.g., scalar + array → array).

    Precision model:
        - Forward: combines means and precisions via precision-weighted addition.
        - Backward: subtracts known input from output and refines residual belief.

    Supports:
        - Real or complex dtype inputs
        - Mixed precision propagation with optional scalar reduction

    Used in:
        - Signal composition models
        - Residual decomposition
        - Summation constraints in factor graphs
    """


    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Combine two input beliefs into a posterior for Z = A + B.

        Args:
            inputs (dict): Contains "a" and "b" UncertainArray inputs.

        Returns:
            UncertainArray: Fused belief for z with updated mean and precision.
    
        Raises:
            RuntimeError: If any input message is missing.
        """

        a = inputs.get("a")
        b = inputs.get("b")

        if a is None or b is None:
            raise RuntimeError("Missing input messages: a or b is None.")

        a, b = a.astype(self.dtype), b.astype(self.dtype)

        mu = a.data + b.data
        precision = 1.0 / (1.0 / a.precision(raw = True) + 1.0 / b.precision(raw = True))

        return UA(mu, dtype=self.dtype, precision=precision)


    def _compute_backward(self, output: UA, exclude: str) -> UA:
        """
        Compute the residual message to one input by subtracting the other.

        Args:
            output: Current belief of z.
            exclude: Which input to exclude when computing the backward message ("a" or "b").

        Returns:
            UncertainArray: The updated message to send to the excluded input.

        Raises:
            RuntimeError: If required inputs are not available.
        """

        other_name = "b" if exclude == "a" else "a"
        other_wave = self.inputs.get(other_name)
        other_msg = self.input_messages.get(other_wave)

        if other_wave is None:
            raise RuntimeError(f"Input wave '{other_name}' not found.")
        if other_msg is None:
            raise RuntimeError(f"Missing message from input wave '{other_name}'.")

        mu = output.data - other_msg.data
        precision = 1.0 / (1.0 / output.precision(raw = True) + 1.0 / other_msg.precision(raw = True))
        out_dtype = get_lower_precision_dtype(output.dtype, other_msg.dtype)
        msg = UA(mu, dtype=out_dtype, precision=precision)

        mode = self.precision_mode_enum

        # Handle scalar projection if mixed precision
        if mode == BPM.SCALAR_AND_ARRAY_TO_ARRAY and exclude == "a":
            a_wave = self.inputs.get("a")
            a_msg = self.input_messages.get(a_wave)
            if a_msg is None:
                raise RuntimeError("Missing input message from wave 'a'.")
            return (a_msg.as_array_precision() * msg).as_scalar_precision() / a_msg

        if mode == BPM.ARRAY_AND_SCALAR_TO_ARRAY and exclude == "b":
            b_wave = self.inputs.get("b")
            b_msg = self.input_messages.get(b_wave)
            if b_msg is None:
                raise RuntimeError("Missing input message from wave 'b'.")
            return (b_msg.as_array_precision() * msg).as_scalar_precision() / b_msg

        return msg

    def get_sample_for_output(self, rng):
        a = self.inputs["a"].get_sample()
        b = self.inputs["b"].get_sample()
        if a is None or b is None:
            raise RuntimeError("Input sample(s) not set for AddPropagator.")
        return a + b

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode_enum.value if self._precision_mode is not None else None
        return f"Add(gen={gen}, mode={mode})"
