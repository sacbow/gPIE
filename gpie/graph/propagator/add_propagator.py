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


    def _compute_forward(
        self,
        inputs: dict[str, UA],
        block: slice | None = None
    ) -> UA:
        """
        Block-aware forward message for Z = A + B.
        """

        a = inputs.get("a")
        b = inputs.get("b")

        if a is None or b is None:
            raise RuntimeError("Missing input messages: a or b is None.")

        # ------------------------------------------------------------
        # Extract block (or full batch if block is None)
        # ------------------------------------------------------------
        a_blk = a.extract_block(block).astype(self.dtype)
        b_blk = b.extract_block(block).astype(self.dtype)

        # ------------------------------------------------------------
        # Gaussian fusion for addition
        # ------------------------------------------------------------
        mu = a_blk.data + b_blk.data
        prec = 1.0 / (
            1.0 / a_blk.precision(raw=True) +
            1.0 / b_blk.precision(raw=True)
        )
        msg = UA(mu, dtype=self.dtype, precision=prec)

        # ------------------------------------------------------------
        # Precision-mode specific projection
        # ------------------------------------------------------------
        mode = self.precision_mode_enum
        if mode == BPM.ARRAY_AND_ARRAY_TO_SCALAR:
            if self.output_message is None:
                return msg.as_scalar_precision()

            # Use full output_message but restrict to block
            out_blk = self.output_message.extract_block(block)
            proj = (msg * out_blk.as_array_precision()).as_scalar_precision()
            return proj / out_blk

        return msg


    def _compute_backward(
        self,
        output: UA,
        exclude: str,
        block: slice | None = None
    ) -> UA:
        """
        Block-aware backward message for Z = A + B.
        """

        other_name = "b" if exclude == "a" else "a"
        other_wave = self.inputs.get(other_name)
        other_msg = self.input_messages.get(other_wave)

        if other_wave is None:
            raise RuntimeError(f"Input wave '{other_name}' not found.")
        if other_msg is None:
            raise RuntimeError(f"Missing message from input wave '{other_name}'.")

        # ------------------------------------------------------------
        # Extract block
        # ------------------------------------------------------------
        z_blk = output.extract_block(block)
        other_blk = other_msg.extract_block(block)

        # ------------------------------------------------------------
        # Residual computation
        # ------------------------------------------------------------
        mu = z_blk.data - other_blk.data
        precision = 1.0 / (
            1.0 / z_blk.precision(raw=True) +
            1.0 / other_blk.precision(raw=True)
        )

        out_dtype = get_lower_precision_dtype(z_blk.dtype, other_blk.dtype)
        msg = UA(mu, dtype=out_dtype, precision=precision)

        # ------------------------------------------------------------
        # Precision-mode specific projection
        # ------------------------------------------------------------
        mode = self.precision_mode_enum

        if mode == BPM.SCALAR_AND_ARRAY_TO_ARRAY and exclude == "a":
            a_wave = self.inputs.get("a")
            a_msg = self.input_messages.get(a_wave)
            if a_msg is None:
                raise RuntimeError("Missing input message from wave 'a'.")

            a_blk = a_msg.extract_block(block)
            return (a_blk.as_array_precision() * msg).as_scalar_precision() / a_blk

        if mode == BPM.ARRAY_AND_SCALAR_TO_ARRAY and exclude == "b":
            b_wave = self.inputs.get("b")
            b_msg = self.input_messages.get(b_wave)
            if b_msg is None:
                raise RuntimeError("Missing input message from wave 'b'.")

            b_blk = b_msg.extract_block(block)
            return (b_blk.as_array_precision() * msg).as_scalar_precision() / b_blk

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
