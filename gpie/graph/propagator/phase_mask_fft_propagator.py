from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.backend import np, move_array_to_current_backend
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_complex_dtype
from ...core.linalg_utils import reduce_precision_to_scalar
from ...core.fft import get_fft_backend


class PhaseMaskFFTPropagator(Propagator):
    """
    Applies a complex-valued phase mask in the frequency domain:
        y = IFFT2( phase_mask * FFT2(x) )

    Args:
        phase_mask (ndarray): Complex array of unit magnitude with shape (H, W).
        precision_mode (UnaryPropagatorPrecisionMode): Scalar/array propagation model.
        dtype (dtype): Complex type (default: complex64).
    """

    def __init__(
        self,
        phase_mask,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype = np().complex64
    ):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)

        if phase_mask.ndim == 2:
            self.mask_needs_batch = True
        elif phase_mask.ndim == 3:
            self.mask_needs_batch = False
        else:
            raise ValueError("phase_mask must be 2D or 3D.")

        if not np().allclose(np().abs(phase_mask), 1.0, atol=1e-6):
            raise ValueError("phase_mask must be unit-magnitude.")

        self.phase_mask = phase_mask
        self.phase_mask_conj = phase_mask.conj()
        self.event_shape = phase_mask.shape[-2:]

        self._init_rng = None
        self.x_belief = None
        self.y_belief = None

    def to_backend(self):
        self.phase_mask = move_array_to_current_backend(self.phase_mask, dtype=self.dtype)
        self.phase_mask_conj = self.phase_mask.conj()
        self.dtype = np().dtype(self.dtype)

    def _set_precision_mode(self, mode: str | UnaryPropagatorPrecisionMode):
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)
        allowed = {
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode: {mode}")
        if hasattr(self, "_precision_mode") and self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(f"Precision mode conflict: {self._precision_mode} vs {mode}")
        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode is not None:
            if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
                return PrecisionMode.ARRAY
            else:
                return PrecisionMode.SCALAR
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode is not None:
            if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
                return PrecisionMode.ARRAY
            else:
                return PrecisionMode.SCALAR
        return None

    def set_precision_mode_forward(self):
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)

    def set_precision_mode_backward(self):
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    def compute_belief(self):
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        msg_y = self.output_message
        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages must be present.")

        if not np().issubdtype(msg_x.data.dtype, np().complexfloating):
            msg_x = msg_x.astype(self.dtype)
        
        fft = get_fft_backend()
        fft2_centered, ifft2_centered = fft.fft2_centered, fft.ifft2_centered

        r = msg_x.data
        p = msg_y.data
        gamma = msg_x._precision
        tau = msg_y._precision
        mode = self._precision_mode

        if mode == UnaryPropagatorPrecisionMode.SCALAR:
            Uh_p = ifft2_centered(self.phase_mask_conj * fft2_centered(p))
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            forward = ifft2_centered(self.phase_mask * fft2_centered(x_mean))
            self.y_belief = UA(forward, dtype=self.dtype, precision=denom)

        elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            Ur = ifft2_centered(self.phase_mask * fft2_centered(r))
            y_mean = (gamma / (gamma + tau)) * Ur + (tau / (gamma + tau)) * p
            self.y_belief = UA(y_mean, dtype=self.dtype, precision=gamma + tau)

            scalar_prec = reduce_precision_to_scalar(gamma + tau)
            back = ifft2_centered(self.phase_mask_conj * fft2_centered(y_mean))
            self.x_belief = UA(back, dtype=self.dtype, precision=scalar_prec)

        elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            Uh_p = ifft2_centered(self.phase_mask_conj * fft2_centered(p))
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)

            scalar_prec = reduce_precision_to_scalar(denom)
            forward = ifft2_centered(self.phase_mask * fft2_centered(x_mean))
            self.y_belief = UA(forward, dtype=self.dtype, precision=scalar_prec)

        else:
            raise ValueError(f"Unknown precision_mode: {mode}")

    def forward(self):
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        out_msg = self.output_message
        yb = self.y_belief

        # --- Case A: initial iteration (no belief messages yet) ---
        if out_msg is None and yb is None:
            if msg_x is None:
                raise RuntimeError(
                    "PhaseMaskFFTPropagator.forward(): missing input message on the initial iteration. "
                    "Upstream prior (or previous node) must emit an initial message before PM-FFT."
                )

            # Ensure complex dtype for the transform path
            if not np().issubdtype(msg_x.data.dtype, np().complexfloating):
                msg_x = msg_x.astype(self.dtype)

            # Transform to frequency domain, apply phase mask, then inverse-transform.
            # We intentionally use UA helpers to preserve the Gaussian-EP semantics
            # (fft2_centered() returns scalar precision; ifft2_centered() keeps scalar precision).
            u = msg_x.fft2_centered()  # UA (scalar precision)
            masked = UA(u.data * self.phase_mask, dtype=self.dtype, precision=u.precision(raw=True))
            msg = masked.ifft2_centered()  # UA (scalar precision)

            # Align precision mode with the output wave (ARRAY or SCALAR)
            if self.output.precision_mode_enum == PrecisionMode.ARRAY:
                msg = msg.as_array_precision()

            self.output.receive_message(self, msg)
            return

        # --- Case B: steady-state EP update (both present) ---
        if out_msg is not None and yb is not None:
            msg = yb / out_msg
            self.output.receive_message(self, msg)
            return

        # --- Case C: inconsistent state ---
        raise RuntimeError(
            "PhaseMaskFFTPropagator.forward(): inconsistent state. "
            "Expected both y_belief and output_message to be None (initial) or both present (update)."
        )


    def backward(self):
        x_wave = self.inputs["input"]
        if self.output_message is None:
            raise RuntimeError("Output message missing.")
        self.compute_belief()
        incoming = self.input_messages[x_wave]
        msg = self.x_belief / incoming
        x_wave.receive_message(self, msg)

    def set_init_rng(self, rng):
        self._init_rng = rng

    def get_sample_for_output(self, rng):
        fft = get_fft_backend()
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        return fft.ifft2_centered(self.phase_mask * fft.fft2_centered(x))

    def __matmul__(self, wave: Wave) -> Wave:
        if len(wave.event_shape) != 2:
            raise ValueError("PhaseMaskFFTPropagator expects 2D wave input.")

        if wave.event_shape != self.event_shape:
            raise ValueError("Input wave shape does not match phase mask shape.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = get_complex_dtype(wave.dtype)

        # Expand or validate phase_mask batch dimension
        if self.mask_needs_batch:
            B = wave.batch_size
            self.phase_mask = np().broadcast_to(self.phase_mask, (B, *self.event_shape))
            self.phase_mask_conj = self.phase_mask.conj()
            self.mask_needs_batch = False
        else:
            if self.phase_mask.shape[0] != wave.batch_size:
                raise ValueError(
                    f"Batch size mismatch: phase_mask batch={self.phase_mask.shape[0]}, wave batch={wave.batch_size}"
                )
        
        if self.phase_mask.shape[1:] != wave.event_shape:
            raise ValueError(
                f"phase_mask event_shape {self.phase_mask.shape[1:]} does not match wave event_shape {wave.event_shape}"
            )

        out = Wave(event_shape=self.event_shape, batch_size=wave.batch_size, dtype=self.dtype)
        out._set_generation(self._generation + 1)
        out.set_parent(self)
        self.output = out
        return self.output

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PMFFTProp(gen={gen}, mode={self.precision_mode})"
