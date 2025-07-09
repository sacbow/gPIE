import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA


class SupportPrior(Prior):
    def __init__(self, support: np.ndarray, dtype=np.complex128, precision_mode: str = "array"):
        """
        Support-based prior. CN(0,1) on support=True, delta(0) on support=False.

        Args:
            support (np.ndarray): Boolean mask indicating support region.
            dtype (np.dtype): Data type of the prior (default: complex128).
            precision_mode (str): 'array' (default) or 'scalar' — scalar is not recommended.
        """
        if support.dtype != bool:
            raise ValueError("Support must be a boolean numpy array.")

        if precision_mode not in ("array", "scalar"):
            raise ValueError("precision_mode must be 'array' or 'scalar'.")

        self.support = support
        self.large_value = 1e6
        self._fixed_msg_array = self._create_fixed_array(dtype)

        super().__init__(shape=support.shape, dtype=dtype, precision_mode=precision_mode)

    def _create_fixed_array(self, dtype):
        mean = np.zeros_like(self.support, dtype=dtype)
        precision = np.where(self.support, 1.0, self.large_value)
        return UA(mean, dtype=dtype, precision=precision)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the prior message.

        - If array mode: return precomputed fixed array.
        - If scalar mode: combine with incoming and reduce.
        """
        if self.output.precision_mode == "array":
            return self._fixed_msg_array
        else:
            # Scalar mode: combine, reduce, divide
            combined = self._fixed_msg_array * incoming
            reduced = combined.as_scalar_precision()
            return reduced / incoming

    def generate_sample(self, rng):
        """
        Generate a sample with zeros outside support, and CN(0,1) inside.
        """
        sample = np.zeros(self.support.shape, dtype=self.dtype)
        n = np.count_nonzero(self.support)
        values = rng.normal(size=n) + 1j * rng.normal(size=n)
        sample[self.support] = values / np.sqrt(2)
        self.output.set_sample(sample)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"SupportPrior(gen={gen}, mode={mode})"