import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import complex_normal_random_array
from typing import Optional


class GaussianPrior(Prior):
    def __init__(
        self,
        mean=0.0,
        var=1.0,
        shape=(1,),
        dtype=np.complex128,
        precision_mode: Optional[str] = None,
        label = None
    ):
        """
        Gaussian prior: each variable follows CN(mean, var) independently.

        Args:
            mean (float or complex): Mean of the complex normal distribution.
            var (float): Variance (scalar).
            shape (tuple): Shape of the variable.
            dtype (np.dtype): Data type.
            precision_mode (str or None): "scalar", "array", or None.
        """
        self.mean = mean
        self.var = var
        self.precision = 1.0 / var

        super().__init__(shape=shape, dtype=dtype, precision_mode=precision_mode, label = label)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the prior message as a constant UncertainArray with fixed precision.
        """
        mode = self.output.precision_mode
        if mode == "scalar":
            return UA.zeros(self.shape, dtype=self.dtype, precision=self.precision)
        elif mode == "array":
            precision_array = np.full(self.shape, self.precision, dtype=np.float64)
            return UA.zeros(self.shape, dtype=self.dtype, precision=precision_array)
        else:
            raise RuntimeError("Precision mode not determined for GaussianPrior output.")

    def generate_sample(self, rng):
        """
        Generate a sample from CN(mean, var) and set it to the output Wave.
        """
        sample = complex_normal_random_array(self.shape, dtype=self.dtype, rng=rng)
        sample = np.sqrt(self.var) * sample + self.mean
        self.output.set_sample(sample)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"GPrior(gen={gen}, mode={self.precision_mode})"
