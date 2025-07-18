import numpy as np
from typing import Optional, Tuple

from .base import Prior
from core.uncertain_array import UncertainArray as UA
from core.types import PrecisionMode
from core.linalg_utils import random_normal_array


class GaussianPrior(Prior):
    def __init__(
        self,
        var: float = 1.0,
        shape: Tuple[int, ...] = (1,),
        dtype: np.dtype = np.complex128,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None
    ) -> None:
        """
        Gaussian prior with zero-mean and scalar variance.
        Supports both real and complex outputs.

        Args:
            var: Variance (must be positive).
            shape: Shape of the latent variable.
            dtype: Output dtype (np.float64 or np.complex128).
            precision_mode: "scalar", "array", or None (inferred).
            label: Optional label for the Wave.
        """
        if var <= 0:
            raise ValueError("Variance must be positive.")
        
        self.var: float = var
        self.precision: float = 1.0 / var

        super().__init__(shape=shape, dtype=dtype, precision_mode=precision_mode, label=label)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the fixed-precision zero-mean Gaussian prior as UncertainArray.
        """
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            return UA.zeros(self.shape, dtype=self.dtype, precision=self.precision, scalar_precision=True)
        elif mode == PrecisionMode.ARRAY:
            return UA.zeros(self.shape, dtype=self.dtype, precision=self.precision, scalar_precision=False)
        else:
            raise RuntimeError("Precision mode not determined for GaussianPrior output.")

    def generate_sample(self, rng: Optional[np.random.Generator]) -> None:
        """
        Generate a sample from N(0, var) or CN(0, var) depending on dtype.
        """
        if rng is None:
            rng = np.random.default_rng()

        sample = np.sqrt(self.var) * random_normal_array(self.shape, dtype=self.dtype, rng=rng)
        self.output.set_sample(sample)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"GaussianPrior(gen={gen}, mode={mode}, var={self.var})"
