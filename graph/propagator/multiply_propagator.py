from __future__ import annotations
from typing import Optional
import numpy as np

from .binary_propagator import BinaryPropagator
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, BinaryPropagatorPrecisionMode as BPMM


class MultiplyPropagator(BinaryPropagator):
    """
    A propagator implementing Z = A * B under complex Gaussian belief propagation.

    This module supports multiplicative interactions between two latent variables
    within the Expectation Propagation (EP) framework.

    Unlike additive propagators, the multiplication is nonlinear and leads to
    non-Gaussian true posteriors. This module approximates them as Gaussians
    by moment matching (mean & variance).

    Precision modes:
        - Supported: ARRAY, SCALAR_AND_ARRAY_TO_ARRAY, ARRAY_AND_SCALAR_TO_ARRAY
        - Not supported: SCALAR (this is a bad approximation)

    Key operations:
        - Forward: Combines beliefs of A and B to estimate Z ≈ A * B
        - Backward: Sends messages to A or B by conditioning on Z and the other

    Note:
        Belief estimates are required on both inputs before forward propagation.

    Typical use cases:
        - Gain-modulated signal modeling
        - Elementwise multiplicative interaction (e.g., masks, amplitude scaling)
    """


    def __init__(self, precision_mode: Optional[BPMM] = None):
        super().__init__(precision_mode=precision_mode)
        self._init_rng = None

    def _set_precision_mode(self, mode: BPMM) -> None:
        allowed = {
            BPMM.ARRAY,
            BPMM.SCALAR_AND_ARRAY_TO_ARRAY,
            BPMM.ARRAY_AND_SCALAR_TO_ARRAY,
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for MultiplyPropagator: '{mode}'")
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict in MultiplyPropagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        a_mode = self.inputs["a"].precision_mode_enum
        b_mode = self.inputs["b"].precision_mode_enum

        if a_mode == PrecisionMode.SCALAR and b_mode == PrecisionMode.SCALAR:
            raise ValueError("MultiplyPropagator does not support scalar × scalar mode.")

        if a_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPMM.SCALAR_AND_ARRAY_TO_ARRAY)
        elif b_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPMM.ARRAY_AND_SCALAR_TO_ARRAY)
        else:
            self._set_precision_mode(BPMM.ARRAY)

    def get_output_precision_mode(self) -> str:
        return PrecisionMode.ARRAY.value

    def set_precision_mode_backward(self) -> None:
        # No backward propagation of precision required
        pass

    def set_init_rng(self, rng):
        self._init_rng = rng

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Approximate the output belief for Z = A * B using moment-matching.

        Args:
            inputs: Dictionary with input messages ("a", "b").

        Returns:
            UncertainArray: Gaussian approximation of the product distribution.
    
        Raises:
            RuntimeError: If required input beliefs are missing.
        """

        x = self.inputs["a"].belief
        y = self.inputs["b"].belief

        if x is None or y is None:
            raise RuntimeError("Belief not available for forward computation.")

        # Ensure matching dtype
        out_dtype = np.result_type(x.dtype, y.dtype)
        if x.dtype != out_dtype:
            x = x.astype(out_dtype)
        if y.dtype != out_dtype:
            y = y.astype(out_dtype)

        x_m, y_m = x.data, y.data
        sx2 = 1.0 / x._precision
        sy2 = 1.0 / y._precision

        mu = x_m * y_m
        var = (np.abs(x_m) ** 2 + sx2) * (np.abs(y_m) ** 2 + sy2) - np.abs(x_m * y_m) ** 2
        prec = 1.0 / np.maximum(var, 1e-12)

        return UA(mu, dtype=out_dtype, precision=prec)

    def _compute_backward(self, output: UA, exclude: str) -> tuple[UA, UA]:
        """
        Compute the backward message and updated belief for the excluded input.

        Args:
            output: The current output message (belief about Z).
            exclude: Either "a" or "b", indicating which input to update.

        Returns:
            tuple:
                - Message to send to the excluded input
                - Updated belief estimate for that input
    
        Raises:
            RuntimeError: If necessary beliefs or messages are missing.
        """

        z_m, gamma_z = output.data, output._precision
        other_wave = self.inputs["b" if exclude == "a" else "a"]

        if other_wave.belief is None:
            other_wave.compute_belief()
        belief_y = other_wave.belief

        y_q = belief_y.data
        sy2 = 1.0 / belief_y._precision
        abs_y2_plus_var = np.abs(y_q) ** 2 + sy2

        mean_msg = np.conj(y_q) * z_m / abs_y2_plus_var
        prec_msg = gamma_z * abs_y2_plus_var
        msg = UA(mean_msg, dtype=output.dtype, precision=prec_msg)

        target_wave = self.inputs[exclude]
        msg_in = self.input_messages.get(target_wave)

        if self.precision_mode in {BPMM.SCALAR_AND_ARRAY_TO_ARRAY, BPMM.ARRAY_AND_SCALAR_TO_ARRAY}:
            if target_wave.precision_mode_enum == PrecisionMode.SCALAR:
                q_x = (msg * msg_in).as_scalar_precision()
                return q_x / msg_in, q_x

        q_x = msg * msg_in
        return msg, q_x

    def forward(self) -> None:
        """
        Send a message from Z = A * B toward the output wave.

        If input beliefs are not available, initializes a random message.

        Raises:
            RuntimeError: If RNG is not set and inputs are missing.
        """

        z_wave = self.output
        if self.inputs["a"].belief is None or self.inputs["b"].belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            msg = UA.random(z_wave.shape, dtype=self.dtype, rng=self._init_rng, scalar_precision = False)  # precision mode of msg is always "array"
        else:
            belief = self._compute_forward(self.input_messages)
            msg = belief / self.output_message if self.output_message is not None else belief
            z_wave.set_belief(belief)

        z_wave.receive_message(self, msg)

    def backward(self) -> None:
        """
        Send messages to both inputs (A and B) based on the output belief.

        This function uses approximate inversion of the product relation.

        Raises:
            RuntimeError: If the output message is not available.
        """

        if self.output_message is None:
            raise RuntimeError("Output message missing.")

        for exclude in ("a", "b"):
            msg, belief = self._compute_backward(self.output_message, exclude)
            wave = self.inputs[exclude]
            wave.receive_message(self, msg)
            wave.set_belief(belief)

    def generate_sample(self, rng):
        a = self.inputs["a"].get_sample()
        b = self.inputs["b"].get_sample()
        if a is None or b is None:
            raise RuntimeError("Input sample(s) not set for MultiplyPropagator.")
        self.output.set_sample(a * b)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"Mul(gen={gen}, mode={self.precision_mode})"
