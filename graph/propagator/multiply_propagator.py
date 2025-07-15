from .binary_propagator import BinaryPropagator
from core.uncertain_array import UncertainArray as UA
import numpy as np

class MultiplyPropagator(BinaryPropagator):
    """
    Propagator implementing Z = X * Y for complex Gaussian belief propagation.

    This propagator handles component-wise multiplication under the assumption of
    independent complex Gaussian beliefs over inputs X and Y, and uses Expectation 
    Propagation to approximate the joint posterior over Z.

    Supports scalar, array, and mixed precision modes (e.g., scalar/array to array).
    """
    def __init__(self, dtype=np.complex128, precision_mode=None):
        super().__init__(dtype=dtype, precision_mode=precision_mode)
        self._init_rng = None
    
    def set_init_rng(self, rng):
        """
        Store RNG for initial message generation.

        This RNG is used when input beliefs are missing (e.g., at t=0),
        to send random UncertainArray messages in forward().
        """
        self._init_rng = rng

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Compute forward belief approximation q(z) ≈ CN(x * y, σ²)
        based on the beliefs of inputs "a" and "b".

        Returns:
            UncertainArray: Approximated belief over z.
        """
        x = self.inputs["a"].belief
        y = self.inputs["b"].belief

        if x is None or y is None:
            raise RuntimeError("Belief not available for forward computation.")

        x_m, y_m = x.data, y.data
        sx2 = 1.0 / x._precision
        sy2 = 1.0 / y._precision

        mu = x_m * y_m
        var = (np.abs(x_m)**2 + sx2) * (np.abs(y_m)**2 + sy2) - np.abs(x_m * y_m)**2
        prec = np.reciprocal(np.maximum(var, 1e-12))

        return UA(mu, dtype=self.dtype, precision=prec)

    def _compute_backward(self, output: UA, exclude: str) -> tuple[UA, UA]:
        """
        Compute the message and updated belief for one of the input variables.

        Args:
            output (UncertainArray): Output message (belief over z).
            exclude (str): Which input variable to exclude ("a" or "b").

        Returns:
            Tuple[UncertainArray, UncertainArray]: (message to input, belief over input)
        """
        z_m, gamma_z = output.data, output._precision
        other_wave = self.inputs["b" if exclude == "a" else "a"]
    
        if other_wave.belief is None:
            other_wave.compute_belief()
        belief_y = other_wave.belief

        y_q = belief_y.data
        sy2 = 1.0 / belief_y._precision
        abs_y2_plus_var = np.abs(y_q)**2 + sy2

        mean_msg = np.conj(y_q) * z_m / abs_y2_plus_var
        prec_msg = gamma_z * abs_y2_plus_var #Note : this becomes an array even when the propagator scalar mode
        msg = UA(mean_msg, dtype=self.dtype, precision=prec_msg)

        target_wave = self.inputs[exclude]
        msg_in = self.input_messages.get(target_wave)

        if self.precision_mode in ("scalar/array to scalar", "array/scalar to array", "scalar"):
            if target_wave.precision_mode == "scalar":
                q_x = (msg * msg_in).as_scalar_precision()
                return q_x / msg_in, q_x

        q_x = msg * msg_in
        return msg, q_x

    def forward(self):
        """
        Send message to output wave.

        If input beliefs are unavailable (e.g. first iteration), send a random message.
        Otherwise, compute belief over Z and send q(z) / m(z).
        """
        z_wave = self.output

        if self.inputs["a"].belief is None or self.inputs["b"].belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")

            mode = self.get_output_precision_mode()
            prec = np.ones(z_wave.shape, dtype=np.float64) if mode == "array" else 1.0
            msg = UA.random(z_wave.shape, dtype=self.dtype, rng=self._init_rng, precision=prec)

        else:
            belief = self._compute_forward(self.input_messages)
            if self.get_output_precision_mode() == "array":
                msg = belief / self.output_message if self.output_message is not None else belief
                z_wave.set_belief(belief)
            else:
                msg = belief.as_scalar_precision() / self.output_message  if self.output_message is not None else belief

        z_wave.receive_message(self, msg)

    def backward(self):
        """
        Send messages to input waves using EP-style backward computation.

        Uses output message and the belief of the opposite input to compute message and belief.
        """
        if self.output_message is None:
            raise RuntimeError("Output message missing.")

        for exclude in ("a", "b"):
            msg, belief = self._compute_backward(self.output_message, exclude)
            target_wave = self.inputs[exclude]
            target_wave.receive_message(self, msg)
            target_wave.set_belief(belief)
    

    def generate_sample(self, rng):
        """
        Generate sample for the output wave by multiplying the input samples.

        Args:
            rng: Not used (included for compatibility).
        """
        a = self.inputs["a"].get_sample()
        b = self.inputs["b"].get_sample()

        if a is None or b is None:
            raise RuntimeError("Input sample(s) not set for MultiplyPropagator.")

        self.output.set_sample(a * b)


    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"Mul(gen={gen}, mode={self.precision_mode})"

