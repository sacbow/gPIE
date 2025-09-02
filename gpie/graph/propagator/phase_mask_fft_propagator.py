from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_complex_dtype
from ...core.linalg_utils import fft2_centered, ifft2_centered, reduce_precision_to_scalar


class PhaseMaskFFTPropagator(Propagator):
    """
    Propagator that applies a fixed complex phase mask in the Fourier domain.

    This module performs a two-step transformation:
        y = IFFT2( phase_mask ⊙ FFT2(x) )

    where:
        - FFT2 / IFFT2 are centered 2D Fourier transforms
        - phase_mask is a complex-valued array with unit magnitude
        - ⊙ denotes elementwise multiplication in the frequency domain

    This structure is commonly used in:
        - Coded diffraction pattern modeling
        - Phase retrieval or holography
        - **Angular Spectrum Method** for light propagation modeling

    Precision handling:
        - Supports SCALAR, SCALAR_TO_ARRAY, ARRAY_TO_SCALAR modes
        - Automatically adjusts forward/backward precision consistency

    Notes:
        - Both input and output live in the spatial domain
        - All modulation is performed in the frequency domain

    Args:
        phase_mask (np().ndarray): Complex array of shape (H, W), with unit magnitude.
        precision_mode (UnaryPropagatorPrecisionMode, optional): Desired precision configuration.
        dtype (np().dtype): Data type, typically np().complex128.
    """

    def __init__(
        self,
        phase_mask: np().ndarray,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype: np().dtype = np().complex128
    ):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)
        if phase_mask.ndim != 2:
            raise ValueError("phase_mask must be 2D.")
        if not np().allclose(np().abs(phase_mask), 1.0, atol=1e-6):
            raise ValueError("phase_mask must be unit-magnitude.")
        self.phase_mask = phase_mask
        self.phase_mask_conj = phase_mask.conj()
        self.shape = phase_mask.shape
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None
    
    def to_backend(self):
        import cupy as cp
        current_backend = np()
        
        # Transfer phase mask first
        if isinstance(self.phase_mask, cp.ndarray) and current_backend.__name__ == "numpy":
            self.phase_mask = self.phase_mask.get().astype(self.dtype)
            self.phase_mask_conj = self.phase_mask_conj.get().astype(self.dtype)
        else:
            self.phase_mask = current_backend.asarray(self.phase_mask, dtype=self.dtype)
            self.phase_mask_conj = current_backend.asarray(self.phase_mask_conj, dtype=self.dtype)
        
        # Sync dtype attributes
        self.dtype = current_backend.dtype(self.dtype)

    def _set_precision_mode(self, mode: str | UnaryPropagatorPrecisionMode):
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)
        allowed = {
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for UnitaryPropagator: {mode}")
        if hasattr(self, "_precision_mode") and self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'"
            )
        self._precision_mode = mode

    @property
    def precision_mode_enum(self) -> Optional[UnaryPropagatorPrecisionMode]:
        return self._precision_mode

    @property
    def precision_mode(self) -> Optional[str]:
        return self._precision_mode.value if self._precision_mode else None

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
            raise RuntimeError("Both input and output messages are required.")

        if not np().issubdtype(msg_x.dtype, np().complexfloating):
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
            self.y_belief = UA(
                ifft2_centered(self.phase_mask * fft2_centered(x_mean)),
                dtype=self.dtype,
                precision=denom
            )

        elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            Ur = ifft2_centered(self.phase_mask * fft2_centered(r))
            y_mean = (gamma / (gamma + tau)) * Ur + (tau / (gamma + tau)) * p
            self.y_belief = UA(y_mean, dtype=self.dtype, precision=gamma + tau)

            scalar_prec = reduce_precision_to_scalar(gamma + tau)
            backprop = ifft2_centered(self.phase_mask_conj * fft2_centered(y_mean))
            self.x_belief = UA(backprop, dtype=self.dtype, precision=scalar_prec)

        elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            Uh_p = ifft2_centered(self.phase_mask_conj * fft2_centered(p))
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)

            scalar_prec = reduce_precision_to_scalar(denom)
            forward = ifft2_centered(self.phase_mask * fft2_centered(x_mean))
            self.y_belief = UA(forward, dtype=self.dtype, precision=scalar_prec)

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
    
    def get_sample_for_output(self, rng):
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        return ifft2_centered(self.phase_mask * fft2_centered(x))

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.shape != self.shape:
            raise ValueError("Input wave shape does not match phase mask shape.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = get_complex_dtype(wave.dtype)
        self.phase_mask = self.phase_mask.astype(self.dtype)
        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PMFFTProp(gen={gen}, mode={self.precision_mode})"
