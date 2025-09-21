from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import fft2_centered, ifft2_centered, reduce_precision_to_scalar
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_complex_dtype


class IFFT2DPropagator(Propagator):
    """
    A centered 2D inverse FFT-based propagator for EP message passing.

    It defines a unitary mapping:
        y = IFFT2_centered(x)
        x = FFT2_centered(y)

    Supports:
        - SCALAR <-> SCALAR
        - SCALAR <-> ARRAY
        - ARRAY -> SCALAR

    Precision handling follows `UnaryPropagatorPrecisionMode`.

    Notes:
        - Assumes event_shape is 2D (e.g., (H, W))
        - Internally uses fftshifted IFFT/FFT
    """

    def __init__(
        self,
        event_shape: Optional[tuple[int, int]] = None,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype=np().complex64,
    ):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)
        self.event_shape = event_shape
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None

    def to_backend(self):
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
            raise ValueError(f"Invalid precision_mode for IFFT2DPropagator: {mode}")
        if hasattr(self, "_precision_mode") and self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'"
            )
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
            raise RuntimeError("Both input and output messages are required to compute belief.")

        if not np().issubdtype(msg_x.data.dtype, np().complexfloating):
            msg_x = msg_x.astype(self.dtype)

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
        return ifft2_centered(x)

    def __matmul__(self, wave: Wave) -> Wave:
        if len(wave.event_shape) != 2:
            raise ValueError(f"IFFT2DPropagator only supports 2D input. Got {wave.event_shape}")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = get_complex_dtype(wave.dtype)
        self.event_shape = wave.event_shape

        out_wave = Wave(event_shape=self.event_shape, batch_size=wave.batch_size, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"IFFT2DProp(gen={gen}, mode={self.precision_mode})"
