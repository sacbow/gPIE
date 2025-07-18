import numpy as np
from typing import Optional
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA
from core.types import PrecisionMode
from core.linalg_utils import fft2_centered, ifft2_centered, reduce_precision_to_scalar


class PhaseMaskFFTPropagator(Propagator):
    def __init__(
        self,
        phase_mask: np.ndarray,
        precision_mode: Optional[str | PrecisionMode] = None,
        dtype: np.dtype = np.complex128
    ):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)
        self.phase_mask = phase_mask
        self.phase_mask_conj = phase_mask.conj()
        self.shape = phase_mask.shape
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None

    def _set_precision_mode(self, mode: str | PrecisionMode) -> None:
        if isinstance(mode, str):
            mode = PrecisionMode(mode)

        allowed = {
            PrecisionMode.SCALAR,
            PrecisionMode.SCALAR_TO_ARRAY,
            PrecisionMode.ARRAY_TO_SCALAR,
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for PhaseMaskFFTPropagator: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'")

        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode == PrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        elif self._precision_mode == PrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.SCALAR
        elif self._precision_mode == PrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.ARRAY
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode in (PrecisionMode.SCALAR, PrecisionMode.ARRAY_TO_SCALAR):
            return PrecisionMode.SCALAR
        elif self._precision_mode == PrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.ARRAY
        return None

    def set_precision_mode_forward(self):
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(PrecisionMode.ARRAY_TO_SCALAR)

    def set_precision_mode_backward(self):
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(PrecisionMode.SCALAR_TO_ARRAY)

    def compute_belief(self):
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required.")

        # Cast input to complex if necessary
        if not np.issubdtype(msg_x.dtype, np.complexfloating):
            msg_x = msg_x.astype(self.dtype)

        r = msg_x.data
        p = msg_y.data
        gamma = msg_x._precision
        tau = msg_y._precision

        if self._precision_mode == PrecisionMode.SCALAR:
            Uh_p = ifft2_centered(self.phase_mask_conj * fft2_centered(p))
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            self.y_belief = UA(
                ifft2_centered(self.phase_mask * fft2_centered(x_mean)),
                dtype=self.dtype,
                precision=denom
            )

        elif self._precision_mode == PrecisionMode.SCALAR_TO_ARRAY:
            Ur = ifft2_centered(self.phase_mask * fft2_centered(r))
            y_mean = (gamma / (gamma + tau)) * Ur + (tau / (gamma + tau)) * p
            self.y_belief = UA(y_mean, dtype=self.dtype, precision=gamma + tau)

            scalar_prec = reduce_precision_to_scalar(gamma + tau)
            backprop = ifft2_centered(self.phase_mask_conj * fft2_centered(y_mean))
            self.x_belief = UA(backprop, dtype=self.dtype, precision=scalar_prec)

        elif self._precision_mode == PrecisionMode.ARRAY_TO_SCALAR:
            Uh_p = ifft2_centered(self.phase_mask_conj * fft2_centered(p))
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)

            scalar_prec = reduce_precision_to_scalar(denom)
            forward = ifft2_centered(self.phase_mask * fft2_centered(x_mean))
            self.y_belief = UA(ifft2_centered(forward), dtype=self.dtype, precision=scalar_prec)

        else:
            raise ValueError(f"Unknown precision_mode: {self._precision_mode}")

    def forward(self):
        if self.output_message is None or self.y_belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            msg = UA.random(self.shape, dtype=self.dtype, rng=self._init_rng)
        else:
            msg = self.y_belief / self.output_message

        self.output.receive_message(self, msg)

    def backward(self):
        x_wave = self.inputs["input"]
        if self.output_message is None:
            raise RuntimeError("Output message missing.")
        self.compute_belief()
        incoming = self.input_messages[x_wave]
        msg_in = self.x_belief / incoming
        x_wave.receive_message(self, msg_in)

    def set_init_rng(self, rng):
        self._init_rng = rng

    def generate_sample(self, rng):
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        y = ifft2_centered(self.phase_mask * fft2_centered(x))
        self.output.set_sample(y)

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.ndim != 2:
            raise ValueError("PhaseMaskFFTPropagator only supports 2D wave input.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.shape = wave.shape
        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PMFFTProp(gen={gen}, mode={self.precision_mode})"