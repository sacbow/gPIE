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
        Perform inner-loop VMP updates for Q_x, Q_y and compute Q_z.
        Also stores raw backward messages to inputs to avoid UA division shrinking.
        """
        qx = self.input_beliefs["a"]
        qy = self.input_beliefs["b"]
        z_msg = self.output_message

        z_m, gamma_z = z_msg.data, z_msg.precision(raw=False)

        # Keep last messages for backward sending
        last_msg_to_x = None
        last_msg_to_y = None

        for _ in range(self.num_inner_loop):
            # --- Q_x update ---
            y_m, sy2 = qy.data, 1.0 / qy.precision(raw=False)
            abs_y2_plus_var = np().abs(y_m) ** 2 + sy2
            mean_x = np().conj(y_m) * z_m / abs_y2_plus_var
            prec_x = gamma_z * abs_y2_plus_var
            msg_to_x = UA(mean_x, dtype=self.dtype, precision=prec_x)  # <-- raw VMP message
            self.input_beliefs["a"] = msg_to_x * self.input_messages[self.inputs["a"]].as_array_precision()
            qx = self.input_beliefs["a"]
            last_msg_to_x = msg_to_x  # <-- store raw message

            # --- Q_y update ---
            x_m, sx2 = qx.data, 1.0 / qx.precision(raw=False)
            abs_x2_plus_var = np().abs(x_m) ** 2 + sx2
            mean_y = np().conj(x_m) * z_m / abs_x2_plus_var
            prec_y = gamma_z * abs_x2_plus_var
            msg_to_y = UA(mean_y, dtype=self.dtype, precision=prec_y)  # <-- raw VMP message
            self.input_beliefs["b"] = msg_to_y * self.input_messages[self.inputs["b"]].as_array_precision()
            qy = self.input_beliefs["b"]
            last_msg_to_y = msg_to_y  # <-- store raw message

        # Save for backward() to send directly (no UA division)
        self._last_backward_msgs = {"a": last_msg_to_x, "b": last_msg_to_y}

        # --- Compute Q_z after inner loop (with eps stabilization) ---
        mu_z = qx.data * qy.data
        var_z = (np().abs(qx.data) ** 2 + 1.0 / qx.precision(raw=False)) * \
                (np().abs(qy.data) ** 2 + 1.0 / qy.precision(raw=False)) - np().abs(mu_z) ** 2
        eps = np().array(1e-8, dtype=get_real_dtype(self.dtype))
        prec_z = 1.0 / np().maximum(var_z, eps)
        self.output_belief = UA(mu_z, dtype=self.dtype, precision=prec_z)



    def forward(self) -> None:
        """
        Deterministic forward message passing for MultiplyPropagator.

        Logic:
        - On the first iteration (no output_belief / output_message):
            → Initialize input_beliefs from input_messages if missing.
            → Use those beliefs (qa, qb) to moment-match Z = X * Y.
            → Construct deterministic UA for output message.
            → Align precision with output wave.
        - On subsequent iterations:
            → Perform EP-style message update using existing beliefs/messages.
        """
        z_wave = self.output
        a_wave = self.inputs["a"]
        b_wave = self.inputs["b"]

        # --- Ensure input_beliefs are initialized from input_messages ---
        if self.input_beliefs["a"] is None:
            self.input_beliefs["a"] = self.input_messages.get(a_wave)
        if self.input_beliefs["b"] is None:
            self.input_beliefs["b"] = self.input_messages.get(b_wave)

        qa = self.input_beliefs["a"]
        qb = self.input_beliefs["b"]
        out_msg = self.output_message
        out_belief = self.output_belief

        # --- Case A: initial iteration (no output belief/message yet) ---
        if out_msg is None and out_belief is None:
            # Moment-matching approximation for Z = X * Y
            mu_z = qa.data * qb.data
            var_z = (
                (np().abs(qa.data) ** 2 + 1.0 / qa.precision(raw=False))
                * (np().abs(qb.data) ** 2 + 1.0 / qb.precision(raw=False))
                - np().abs(mu_z) ** 2
            )
            eps = np().array(1e-8, dtype=get_real_dtype(self.dtype))
            prec_z = 1.0 / np().maximum(var_z, eps)

            msg = UA(mu_z, dtype=self.dtype, precision=prec_z)

            # Align precision mode with output wave
            if z_wave.precision_mode_enum == PrecisionMode.SCALAR:
                msg = msg.as_scalar_precision()
            else:
                msg = msg.as_array_precision()

            # Save as output_belief and emit to output wave
            self.output_belief = msg
            z_wave.receive_message(self, msg)
            return

        # --- Case B: subsequent EP-style iteration ---
        if out_msg is not None and out_belief is not None:
            if z_wave.precision_mode_enum == PrecisionMode.SCALAR:
                msg = out_belief.as_scalar_precision() / out_msg
            else:
                msg = out_belief / out_msg
            z_wave.receive_message(self, msg)
            return

        # --- Case C: inconsistent message state ---
        raise RuntimeError(
            "MultiplyPropagator.forward(): inconsistent state — "
            "expected both output_belief/output_message to be None (init) or both present (EP update)."
        )


    def backward(self) -> None:
        """
        Run VMP updates and send messages to inputs.
        - For array-precision waves: send raw VMP message (no UA division)
        - For scalar-precision waves: send scalarized belief divided by incoming message
        """
        if self.output_message is None:
            raise RuntimeError("Output message missing.")
        if any(v is None for v in self.input_beliefs.values()):
            raise RuntimeError("Input beliefs not initialized before backward().")

        # Perform VMP updates (stores _last_backward_msgs and updates input_beliefs)
        self.compute_variational_inference()

        for name, wave in self.inputs.items():
            msg_in = self.input_messages[wave]
            belief = self.input_beliefs[name]

            if wave.precision_mode_enum == PrecisionMode.SCALAR:
                # Scalar precision → use scalarized belief / incoming message
                msg = belief.as_scalar_precision() / msg_in
            else:
                # Array precision → send raw VMP message as-is
                msg = self._last_backward_msgs[name]

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
