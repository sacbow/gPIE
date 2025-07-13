from .binary_propagator import BinaryPropagator
from core.uncertain_array import UncertainArray as UA


class AddPropagator(BinaryPropagator):
    """
    A propagator that models Z = A + B under Gaussian belief propagation.

    Given two independent Gaussian inputs (A, B), this propagator computes the belief for 
    the output variable Z. It supports multiple precision modes, including scalar and array,
    and handles backward message computation with optional scalar projection depending on mode.

    The computation is based on the assumption of independent additive Gaussian variables:
        μ_z = μ_a + μ_b
        γ_z = 1 / (1/γ_a + 1/γ_b)
    """

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Compute the forward message from inputs A and B to output Z.

        Args:
            inputs (dict[str, UA]): Messages from input waves "a" and "b".

        Returns:
            UA: The resulting message to the output wave Z.
        """
        a = inputs.get("a")
        b = inputs.get("b")

        if a is None or b is None:
            raise RuntimeError("Missing input messages: a or b is None.")

        mu = a.data + b.data
        precision = 1.0 / (1.0 / a._precision + 1.0 / b._precision)

        return UA(mu, dtype=a.dtype, precision=precision)

    def _compute_backward(self, output: UA, exclude: str) -> UA:
        """
        Compute the backward message from output Z to input A or B.

        The message is computed using Gaussian subtraction. In mixed precision modes
        ("scalar/array to array", "array/scalar to array"), scalar projection is applied 
        to the product of the belief and the computed message before division.

        Args:
            output (UA): The message from the output wave Z.
            exclude (str): Which input ("a" or "b") the message should be sent to.

        Returns:
            UA: The message to the specified input wave.
        """
        other_name = "b" if exclude == "a" else "a"
        other_wave = self.inputs.get(other_name)
        other_msg = self.input_messages.get(other_wave)

        if other_wave is None:
            raise RuntimeError(f"Input wave '{other_name}' not found.")

        if other_msg is None:
            raise RuntimeError(f"Missing message from input wave '{other_name}'.")

        mu = output.data - other_msg.data
        precision = 1.0 / (1.0 / output._precision + 1.0 / other_msg._precision)
        msg = UA(mu, dtype=output.dtype, precision=precision)

        mode = self.precision_mode

        if mode == "scalar/array to array" and exclude == "a":
            a_wave = self.inputs.get("a")
            a_msg = self.input_messages.get(a_wave)
            if a_msg is None:
                raise RuntimeError("Missing input message from wave 'a'.")
            tmp = a_msg * msg
            return tmp.as_scalar_precision() / a_msg

        if mode == "array/scalar to array" and exclude == "b":
            b_wave = self.inputs.get("b")
            b_msg = self.input_messages.get(b_wave)
            if b_msg is None:
                raise RuntimeError("Missing input message from wave 'b'.")
            tmp = b_msg * msg
            return tmp.as_scalar_precision() / b_msg

        return msg
    
    def generate_sample(self, rng):
        """
        Generate sample for the output wave by summing the input samples.

        Args:
            rng: Not used (included for compatibility).
        """
        a = self.inputs["a"].get_sample()
        b = self.inputs["b"].get_sample()

        if a is None or b is None:
            raise RuntimeError("Input sample(s) not set for AddPropagator.")

        self.output.set_sample(a + b)

    
    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"Add(gen={gen}, mode={self._precision_mode})"
