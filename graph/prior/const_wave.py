import numpy as np
from typing import Optional

from .base import Prior
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode


class ConstWave(Prior):
    def __init__(
        self,
        data: np.ndarray,
        large_value: float = 1e12,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None
    ) -> None:
        """
        Deterministic prior with fixed data and high precision (approximate delta function).

        Args:
            data: Fixed mean values for the wave.
            large_value: Precision value to simulate near-deterministic prior.
            precision_mode: "scalar", "array", or None.
            label: Optional wave label.
        """
        self._data: np.ndarray = np.asarray(data)
        self.large_value: float = large_value

        super().__init__(
            shape=self._data.shape,
            dtype=self._data.dtype,
            precision_mode=precision_mode,
            label=label
        )

    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            return UA(self._data, dtype=self.dtype, precision=self.large_value)
        elif mode == PrecisionMode.ARRAY:
            prec_array = np.full(self._data.shape, self.large_value, dtype=np.float64)
            return UA(self._data, dtype=self.dtype, precision=prec_array)
        else:
            raise RuntimeError("Precision mode not determined for ConstWave output.")

    def generate_sample(self, rng: Optional[np.random.Generator]) -> None:
        self.output.set_sample(self._data)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"ConstWave(gen={gen}, mode={mode})"
