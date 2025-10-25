from __future__ import annotations
from typing import Optional
from .binary_propagator import BinaryPropagator
from ..wave import Wave
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, BinaryPropagatorPrecisionMode as BPM, get_lower_precision_dtype, get_real_dtype


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


    def __init__(self, precision_mode: Optional[BPM] = None, num_inner_loop: int = 2):
        super().__init__(precision_mode=precision_mode)
        # Beliefs for input and output variables
        self.input_beliefs = {"a": None, "b": None}
        self.output_belief: Optional[UA] = None
        # Number of inner-loop updates
        self.num_inner_loop = num_inner_loop


    def _set_precision_mode(self, mode: BPM) -> None:
        allowed = {
            BPM.ARRAY,
            BPM.SCALAR_AND_ARRAY_TO_ARRAY,
            BPM.ARRAY_AND_SCALAR_TO_ARRAY,
            BPM.ARRAY_AND_ARRAY_TO_SCALAR,
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
            self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
        elif b_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
    
    def set_precision_mode_backward(self) -> None:
        z_mode = self.output.precision_mode_enum
        a_mode = self.inputs["a"].precision_mode_enum
        b_mode = self.inputs["b"].precision_mode_enum

        if z_mode is None or z_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(BPM.ARRAY_AND_ARRAY_TO_SCALAR)
            return

        if z_mode == PrecisionMode.ARRAY:
            if a_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
            elif b_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
            else:
                self._set_precision_mode(BPM.ARRAY)


    def compute_variational_inference(self) -> None:
        """
        Perform inner-loop variational updates for Q_x, Q_y, and compute Q_z.
        Assumes:
            - self.input_messages and self.input_beliefs are already populated.
            - self.output_message is available.
        Updates:
            - self.input_beliefs["a"], self.input_beliefs["b"], self.output_belief
        """
        # Local references
        qx = self.input_beliefs["a"]
        qy = self.input_beliefs["b"]
        z_msg = self.output_message

        # Shortcuts
        z_m, gamma_z = z_msg.data, z_msg.precision(raw=False)

        # Iterative update of Q_x and Q_y
        for _ in range(self.num_inner_loop):
            # --- Q_x update ---
            y_m, sy2 = qy.data, 1.0 / qy.precision(raw=False)
            abs_y2_plus_var = np().abs(y_m) ** 2 + sy2
            mean_x = np().conj(y_m) * z_m / abs_y2_plus_var
            prec_x = gamma_z * abs_y2_plus_var
            msg_to_x = UA(mean_x, dtype=self.dtype, precision=prec_x)
            self.input_beliefs["a"] = msg_to_x * self.input_messages[self.inputs["a"]].as_array_precision()
            qx = self.input_beliefs["a"]

            # --- Q_y update ---
            x_m, sx2 = qx.data, 1.0 / qx.precision(raw=False)
            abs_x2_plus_var = np().abs(x_m) ** 2 + sx2
            mean_y = np().conj(x_m) * z_m / abs_x2_plus_var
            prec_y = gamma_z * abs_x2_plus_var
            msg_to_y = UA(mean_y, dtype=self.dtype, precision=prec_y)
            self.input_beliefs["b"] = msg_to_y * self.input_messages[self.inputs["b"]].as_array_precision()
            qy = self.input_beliefs["b"]

        # --- Compute Q_z after inner loop ---
        mu_z = qx.data * qy.data
        var_z = (np().abs(qx.data) ** 2 + 1 / qx.precision(raw=False)) * \
                (np().abs(qy.data) ** 2 + 1 / qy.precision(raw=False)) - np().abs(mu_z) ** 2

        prec_z = 1.0 / var_z

        self.output_belief = UA(mu_z, dtype=self.dtype, precision=prec_z)




    def forward(self) -> None:
        """
        Send message to output wave based on current output_belief.
        If output_belief is not ready, send a random initialization message.
        """
        z_wave = self.output

        # Initialize input beliefs from current input messages if missing
        for name, wave in self.inputs.items():
            if self.input_beliefs[name] is None:
                self.input_beliefs[name] = self.input_messages[wave]

        # If no prior belief, random init message
        if self.output_belief is None or self.output_message is None:
            msg = UA.random(
                event_shape=z_wave.event_shape,
                batch_size=z_wave.batch_size,
                dtype=self.dtype,
                scalar_precision=(z_wave.precision_mode_enum == PrecisionMode.SCALAR),
                rng=self._init_rng,
            )
        else:
            if z_wave.precision_mode_enum == PrecisionMode.SCALAR:
                msg = self.output_belief.as_scalar_precision() / self.output_message
            else:
                msg = self.output_belief / self.output_message

        z_wave.receive_message(self, msg)


    def backward(self) -> None:
        """
        Perform variational inference (update Q_x, Q_y, Q_z) and send messages to inputs.
        """
        if self.output_message is None:
            raise RuntimeError("Output message missing.")

        if any(v is None for v in self.input_beliefs.values()):
            raise RuntimeError("Input beliefs not initialized before backward().")

        # Run inner-loop inference
        self.compute_variational_inference()

        # Send updated messages to inputs (precision-mode aware)
        for name, wave in self.inputs.items():
            belief = self.input_beliefs[name]
            msg_in = self.input_messages[wave]

            if wave.precision_mode_enum == PrecisionMode.SCALAR:
                msg = belief.as_scalar_precision() / msg_in
            else:
                msg = belief / msg_in

            wave.receive_message(self, msg)



    def get_sample_for_output(self, rng):
        a = self.inputs["a"].get_sample()
        b = self.inputs["b"].get_sample()
        if a is None or b is None:
            raise RuntimeError("Input sample(s) not set for MultiplyPropagator.")
        return a * b

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"Mul(gen={gen}, mode={self.precision_mode})"
