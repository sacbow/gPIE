from ...core.backend import np
from typing import Optional, Any

from .base import Prior
from ...core.linalg_utils import random_normal_array
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode


class ConstWave(Prior):
    def __init__(
        self,
        data: Any,
        large_value: float = 1e8,
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
        self._data = data
        self.large_value: float = large_value

        super().__init__(
            shape=self._data.shape,
            dtype=self._data.dtype,
            precision_mode=precision_mode,
            label=label
        )
    
    def to_backend(self) -> None:
        """
        Convert stored data array to current backend (e.g., CuPy or NumPy).
        Ensures backend-safe transfer and dtype synchronization.
        """
        import importlib

        # 現在のbackend名を取得
        backend_name = np().__name__

        if backend_name == "numpy":
            if importlib.util.find_spec("cupy"):
                import cupy as cp
                if isinstance(self._data, cp.ndarray):
                    self._data = self._data.get()
                else:
                    self._data = np().asarray(self._data)
            else:
                self._data = np().asarray(self._data)
        else:
            self._data = np().asarray(self._data)

        self.dtype = self._data.dtype


    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            return UA(self._data, dtype=self.dtype, precision=self.large_value)
        elif mode == PrecisionMode.ARRAY:
            prec_array = np().full(self._data.shape, self.large_value, dtype=np().float64)
            return UA(self._data, dtype=self.dtype, precision=prec_array)
        else:
            raise RuntimeError("Precision mode not determined for ConstWave output.")
    
    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return the fixed sample stored in this constant prior.

        Args:
            rng (Optional): Unused; included for compatibility.

        Returns:
            np().ndarray: The constant data array.
        """
        return self._data + np().sqrt(1.0/self.large_value) * random_normal_array(self.shape, dtype=self.dtype, rng=rng)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"ConstWave(gen={gen}, mode={mode})"
