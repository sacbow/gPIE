import numpy as np
from typing import Optional

from .base import Prior
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_normal_array
from core.types import PrecisionMode


class SupportPrior(Prior):
    def __init__(
        self,
        support: np.ndarray,
        dtype: np.dtype = np.complex128,
        precision_mode: Optional[PrecisionMode] = PrecisionMode.ARRAY,
        label: Optional[str] = None
    ) -> None:
        """
        Support-based prior: CN(0,1) on support=True, delta(0) on support=False.

        Args:
            support: Boolean mask indicating support region.
            dtype: np.float64 or np.complex128.
            precision_mode: Enum value (scalar/array), default: array.
            label: Optional label for the output wave.
        """
        if support.dtype != bool:
            raise ValueError("Support must be a boolean numpy array.")

        self.support: np.ndarray = support
        self.large_value: float = 1e6

        super().__init__(
            shape=support.shape,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label
        )

        self._fixed_msg_array: UA = self._create_fixed_array(dtype)

    def _create_fixed_array(self, dtype: np.dtype) -> UA:
        mean = np.zeros_like(self.support, dtype=dtype)
        precision = np.where(self.support, 1.0, self.large_value)
        return UA(mean, dtype=dtype, precision=precision)

    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.ARRAY:
            return self._fixed_msg_array
        elif mode == PrecisionMode.SCALAR:
            combined = self._fixed_msg_array * incoming
            reduced = combined.as_scalar_precision()
            return reduced / incoming
        else:
            raise RuntimeError("Precision mode not determined for SupportPrior output.")

    def generate_sample(self, rng: Optional[np.random.Generator]) -> None:
        """
        Generate a sample with N(0,1) or CN(0,1) on support=True, and 0 elsewhere.
        """
        if rng is None:
            rng = np.random.default_rng()

        sample = np.zeros(self.support.shape, dtype=self.dtype)
        values = random_normal_array(self.support.shape, dtype=self.dtype, rng=rng)
        sample[self.support] = values[self.support]
        self.output.set_sample(sample)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"SupportPrior(gen={gen}, mode={mode})"
