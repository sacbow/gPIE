import numpy as np
from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import fft2_centered, ifft2_centered, reduce_precision_to_scalar
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode


class IFFT2DPropagator(Propagator):
    """
    Inverse 2D FFT propagator: inverse of `FFT2DPropagator`.

    This propagator applies a centered inverse Fourier transform:
        - x (frequency domain) ⟶ y (spatial domain)

    It mirrors the structure and behavior of `FFT2DPropagator`,
    and supports the same scalar/array precision mode conversions:
        - SCALAR → SCALAR
        - SCALAR → ARRAY
        - ARRAY → SCALAR

    Internally uses `ifft2_centered()` for forward mapping and `fft2_centered()` for reverse.
    """

    def __init__(
        self,
        shape=None,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype=np.complex128
    ):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)
        self.shape = shape
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None

    def _set_precision_mode(self, mode: str | UnaryPropagatorPrecisionMode):
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)
        allowed = {
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for IFFT2DPropagator: {mode}")
        if hasattr(self, "_precision_mode") and self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'")
        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode in (
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
        ):
            return PrecisionMode.SCALAR
        elif self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.ARRAY
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode in (
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
        ):
            return PrecisionMode.SCALAR
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.ARRAY
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
            raise RuntimeError("Both input and output messages are required to compute belief.")

        if not np.issubdtype(msg_x.data.dtype, np.complexfloating):
            msg_x = msg_x.astype(np.complex128)

        r = msg_x.data
        p = msg_y.data
        gamma = msg_x._precision
        tau = msg_y._precision
        mode = self._precision_mode

        if mode == UnaryPropagatorPrecisionMode.SCALAR:
            Uh_p = fft2_centered(p)
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            self.y_belief = UA(ifft2_centered(x_mean), dtype=self.dtype, precision=denom)

        elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            Ur = ifft2_centered(r)
            denom = gamma + tau
            y_mean = (gamma / denom) * Ur + (tau / denom) * p
            self.y_belief = UA(y_mean, dtype=self.dtype, precision=denom)
            scalar_prec = reduce_precision_to_scalar(denom)
            self.x_belief = UA(fft2_centered(y_mean), dtype=self.dtype, precision=scalar_prec)

        elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            Uh_p = fft2_centered(p)
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            scalar_prec = reduce_precision_to_scalar(denom)
            self.y_belief = UA(ifft2_centered(x_mean), dtype=self.dtype, precision=scalar_prec)

        else:
            raise ValueError(f"Unknown precision_mode: {self._precision_mode}")

    def forward(self):
        if self.output_message is None or self.y_belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            if self.output.precision_mode == "scalar":
                msg = UA.random(self.shape, dtype=self.dtype, rng=self._init_rng, scalar_precision = True)
            else:
                msg = UA.random(self.shape, dtype=self.dtype, rng=self._init_rng, scalar_precision = False)
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
        y = ifft2_centered(x)
        self.output.set_sample(y)

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.ndim != 2:
            raise ValueError("IFFT2DPropagator only supports 2D wave input.")
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
        return f"IFFT2DProp(gen={gen}, mode={self.precision_mode})"
