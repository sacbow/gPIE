import numpy as np
from typing import Optional
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA
from core.types import PrecisionMode


class PhaseMaskPropagator(Propagator):
    """
    Component-wise complex phase modulation propagator.
    Applies a fixed unit-magnitude phase mask to the wave.
    """

    def __init__(self, phase_mask, dtype=np.complex128):
        """
        Args:
            phase_mask (ndarray): Complex array of shape (H, W) with unit magnitude.
            dtype (np.dtype): Complex dtype (default: complex128).
        """
        super().__init__(input_names=("input",), dtype=dtype)

        if not np.allclose(np.abs(phase_mask), 1.0):
            raise ValueError("Phase mask must have unit magnitude.")

        self.phase_mask = phase_mask
        self.shape = phase_mask.shape

    def _set_precision_mode(self, mode: str | PrecisionMode) -> None:
        """
        Set the precision mode for this propagator with type-safe handling.

        Args:
            mode: Either a string or a PrecisionMode enum.

        Raises:
            ValueError: If the mode is invalid or conflicts with previously set mode.
        """
        if isinstance(mode, str):
            mode = PrecisionMode(mode)

        if mode.value not in ("scalar", "array"):
            raise ValueError(f"Invalid precision mode for PhaseMaskPropagator: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for PhaseMaskPropagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        return self._precision_mode

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        return self._precision_mode

    def set_precision_mode_forward(self):
        mode = self.inputs["input"].precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode.value)

    def set_precision_mode_backward(self):
        mode = self.output.precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode.value)

    def _compute_forward(self, incoming: dict[str, UA]) -> UA:
        ua = incoming["input"]
        data = ua.data.astype(self.dtype, copy=False)
        result = data * self.phase_mask
        return UA(result, dtype=self.dtype, precision=ua._precision)

    def _compute_backward(self, outgoing: UA, exclude: str = None) -> UA:
        data = outgoing.data.astype(self.dtype, copy=False)
        result = data / self.phase_mask
        return UA(result, dtype=self.dtype, precision=outgoing._precision)

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.shape != self.shape:
            raise ValueError("Input wave shape does not match phase mask shape.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def generate_sample(self, rng):
        x = self.inputs["input"].get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        x = x.astype(self.dtype, copy=False)
        self.output.set_sample(x * self.phase_mask)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PhaseMaskProp(gen={gen}, mode={self.precision_mode})"
