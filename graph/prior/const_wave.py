import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA
from typing import Optional


class ConstWave(Prior):
    def __init__(
        self,
        data: np.ndarray,
        large_value: float = 1e6,
        precision_mode: Optional[str] = None,
        label: Optional[str] = None
    ):
        """
        Deterministic prior with fixed data and high precision (approximate delta function).

        Args:
            data (np.ndarray): Fixed mean values for the wave.
            large_value (float): Precision value to simulate near-deterministic prior.
            precision_mode (str or None): "scalar", "array", or None (auto-inferred).
            label (str or None): Optional label for the output wave.
        """
        self._data = np.asarray(data)
        self.large_value = large_value

        super().__init__(
            shape=self._data.shape,
            dtype=self._data.dtype,
            precision_mode=precision_mode,
            label=label
        )

    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode
        if mode == "scalar":
            return UA(self._data, dtype=self.dtype, precision=self.large_value)
        elif mode == "array":
            prec_array = np.full(self._data.shape, self.large_value, dtype=np.float64)
            return UA(self._data, dtype=self.dtype, precision=prec_array)
        else:
            raise RuntimeError("Precision mode not determined for ConstWave output.")

    def generate_sample(self, rng):
        self.output.set_sample(self._data)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"ConstWave(gen={gen}, mode={self.precision_mode})"
