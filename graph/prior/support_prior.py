import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA

class SupportPrior(Prior):
    def __init__(self, support: np.ndarray, dtype=np.complex128, scalar_precision=False):
        """
        Support-based prior. CN(0,1) on support=True, delta(0) on support=False.

        Args:
            support (np.ndarray): Boolean mask indicating support region.
            dtype (np.dtype): Data type of the prior (default: complex128).
            scalar_precision (bool): Whether to enforce scalar precision in output.
        """
        if support.dtype != bool:
            raise ValueError("Support must be a boolean numpy array.")
        self.support = support
        self.large_value = 1e6  # Used to approximate infinite precision
        self.scalar_precision = scalar_precision
        super().__init__(shape=support.shape, dtype=dtype, scalar_precision=scalar_precision)

        # Precompute fixed message (used when scalar_precision is False)
        mean = np.zeros_like(support, dtype=dtype)
        precision_array = np.where(support, 1.0, self.large_value)
        self.fixed_message = UA(mean, dtype=dtype, precision=precision_array)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return prior message. If scalar_precision=False, always return fixed array.
        If scalar_precision=True, compute belief from incoming and combine, then reduce.
        """
        if not self.scalar_precision:
            return self.fixed_message
        else:
            # Combine prior with incoming
            belief = self.fixed_message * incoming
            belief_scalar = belief.as_scalar_precision()
            return belief_scalar / incoming

    def generate_sample(self, rng):
        """
        Generate a sample with zeros outside support, and CN(0,1) inside.
        """
        shape = self.support.shape
        sample = np.zeros(shape, dtype=self.dtype)
        num_nonzero = np.count_nonzero(self.support)
        values = rng.normal(size=num_nonzero) + 1j * rng.normal(size=num_nonzero)
        sample[self.support] = values / np.sqrt(2)
        self.output.set_sample(sample)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"SupportPrior(gen={gen})"
