import numpy as np
from typing import Optional
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import fft2_centered, ifft2_centered, reduce_precision_to_scalar

class PhaseMaskFFTPropagator(Propagator):
    """
    FFT-based propagator with a fixed complex-valued phase mask in the Fourier domain.

    This operator performs a linear transformation defined as:
        y = IFFT2(phase_mask * FFT2(x))

    Such a transformation corresponds to free-space or near-field propagation modeled
    by the Angular Spectrum Method (ASM), where the phase mask encodes propagation distance.

    It supports scalar and array precision modes, and behaves similarly to
    a unitary transformation from the inference perspective.

    Args:
        phase_mask (np.ndarray): Complex-valued 2D phase mask in Fourier domain.
        precision_mode (str or None): "scalar", "scalar to array", or "array to scalar".
        dtype (np.dtype): Data type of internal variables (default: np.complex128).
    """

    def __init__(self, phase_mask: np.ndarray, precision_mode: Optional[str] = None, dtype=np.complex128):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)
        self.phase_mask = phase_mask  # complex-valued unit-magnitude array
        self.shape = phase_mask.shape
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None

    def _set_precision_mode(self, mode: str):
        allowed = ("scalar", "scalar to array", "array to scalar")
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for PhaseMaskFFTPropagator: {mode}")
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'")
        self._precision_mode = mode

    @property
    def precision_mode(self) -> Optional[str]:
        return self._precision_mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[str]:
        if self._precision_mode in ("scalar", "scalar to array"):
            return "scalar"
        elif self._precision_mode == "array to scalar":
            return "array"
        return None

    def get_output_precision_mode(self) -> Optional[str]:
        if self._precision_mode in ("scalar", "array to scalar"):
            return "scalar"
        elif self._precision_mode == "scalar to array":
            return "array"
        return None

    def set_precision_mode_forward(self):
        x_wave = self.inputs["input"]
        if x_wave.precision_mode == "array":
            self._set_precision_mode("array to scalar")

    def set_precision_mode_backward(self):
        y_wave = self.output
        if y_wave.precision_mode == "array":
            self._set_precision_mode("scalar to array")

    def compute_belief(self):
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required.")

        r = msg_x.data
        p = msg_y.data
        gamma = msg_x._precision
        tau = msg_y._precision

        if self._precision_mode == "scalar":
            Uh_p = ifft2_centered(self.phase_mask * fft2_centered(p))
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            self.y_belief = UA(
                ifft2_centered(self.phase_mask * fft2_centered(x_mean)),
                dtype=self.dtype,
                precision=denom
            )

        elif self._precision_mode == "scalar to array":
            Ur = fft2_centered(r)
            HUr = self.phase_mask * Ur
            y_mean = (gamma / (gamma + tau)) * HUr + (tau / (gamma + tau)) * p
            self.y_belief = UA(y_mean, dtype=self.dtype, precision=gamma + tau)

            scalar_prec = reduce_precision_to_scalar(gamma + tau)
            backprop = ifft2_centered(self.phase_mask * fft2_centered(y_mean))
            self.x_belief = UA(backprop, dtype=self.dtype, precision=scalar_prec)

        elif self._precision_mode == "array to scalar":
            Uh_p = ifft2_centered(self.phase_mask * fft2_centered(p))
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)

            scalar_prec = reduce_precision_to_scalar(denom)
            forward = self.phase_mask * fft2_centered(x_mean)
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
        return f"PMFFTProp(gen={gen}, mode={self._precision_mode})"
