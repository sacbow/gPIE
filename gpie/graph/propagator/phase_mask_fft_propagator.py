from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.backend import np, move_array_to_current_backend
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_complex_dtype
from ...core.linalg_utils import fft2_centered, ifft2_centered, reduce_precision_to_scalar


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
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.ARRAY
        return PrecisionMode.SCALAR

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.ARRAY
        return PrecisionMode.SCALAR

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
        if self.output_message is None or self.y_belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            scalar = self.output.precision_mode_enum == PrecisionMode.SCALAR
            msg = UA.random(
                event_shape=self.event_shape,
                batch_size=self.output.batch_size,
                dtype=self.dtype,
                scalar_precision=scalar,
                rng=self._init_rng,
            )
        else:
            msg = self.y_belief / self.output_message
        self.output.receive_message(self, msg)

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
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        return ifft2_centered(self.phase_mask * fft2_centered(x))

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

        out = Wave(event_shape=self.event_shape, batch_size=wave.batch_size, dtype=self.dtype)
        out._set_generation(self._generation + 1)
        out.set_parent(self)
        self.output = out
        return self.output

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PMFFTProp(gen={gen}, mode={self.precision_mode})"
