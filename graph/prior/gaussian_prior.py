import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA

class GaussianPrior(Prior):
    def __init__(self, mean=0.0, var=1.0, shape=(1,), dtype=np.complex128, seed=None):
        """
        Gaussian prior: each variable follows CN(mean, var) independently.
        For simplicity, only precision is used internally.
        """
        self.mean = mean
        self.var = var
        self.precision = 1.0 / var

        super().__init__(shape=shape, dtype=dtype, seed=seed)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return constant prior message with fixed precision.
        """
        return UA.zeros(self.shape, dtype=self.dtype, precision=self.precision)
    
    def generate_sample(self, rng):
        """
        Generate a sample from CN(mean, var) and set it to the output Wave.
        """
        std = np.sqrt(self.var)
        sample = std * (
            rng.normal(0.0, np.sqrt(0.5), self.shape) +
            1j * rng.normal(0.0, np.sqrt(0.5), self.shape)
        )
        sample += self.mean
        self.output.set_sample(sample.astype(self.dtype))
    
    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"GPrior(gen={gen})"