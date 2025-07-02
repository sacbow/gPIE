import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA


class SupportPrior(Prior):
    def __init__(self, support: np.ndarray, damping: float = 0.0, dtype=np.complex128):
        """
        Prior that enforces a support constraint via a binary mask.

        Args:
            support (ndarray): Boolean array indicating support region (True = allowed).
            damping (float): Optional damping factor [0.0, 1.0].
            dtype (np.dtype): Data type of the wave (complex or real).
        """
        if not np.issubdtype(support.dtype, np.bool_):
            raise ValueError("Support must be a boolean array.")
        if not np.any(support):
            raise RuntimeError("Support mask is entirely False; posterior undefined.")

        super().__init__(shape=support.shape, dtype=dtype)

        self.support = support
        self.damping = damping
        self.old_msg = None

    def _compute_message(self, incoming: UA) -> UA:
        """
        Compute message from prior to wave based on support-constrained posterior.
        Applies damping against previous message if configured.
        """
        belief = self._approximate_posterior(incoming)
        new_msg = belief / incoming

        if self.old_msg is not None and self.damping > 0:
            msg_to_send = new_msg.damp_with(self.old_msg, alpha=self.damping)
        else:
            msg_to_send = new_msg

        self.old_msg = msg_to_send
        return msg_to_send

    def _approximate_posterior(self, incoming: UA) -> UA:
        """
        Apply support constraint to posterior distribution:
        - Inside support: posterior from CN(0,1) prior.
        - Outside support: mean forced to zero.
        Returns posterior with scalar precision (averaged over all pixels).
        """
        m = incoming.data
        v = 1.0 / incoming.precision

        post_var_in = 1.0 / (1.0 + 1.0 / v)
        post_mean_in = post_var_in * (m / v)

        post_mean = np.where(self.support, post_mean_in, 0.0)
        post_var = np.where(self.support, post_var_in, 0.0)
        eps = 1e-12
        avg_var = np.maximum(np.mean(post_var), eps)
        scalar_precision = 1.0 / avg_var

        return UA(post_mean, dtype=self.dtype, precision=scalar_precision)

    def generate_sample(self, rng):
        """
        Generate a sample that is nonzero only within the support region.
        Values inside support are drawn from CN(0,1), outside set to zero.
        """
        sample = np.zeros(self.shape, dtype=self.dtype)

        num_active = np.count_nonzero(self.support)
        real = rng.normal(size=num_active)
        imag = rng.normal(size=num_active)
        values = (real + 1j * imag).astype(self.dtype)

        sample[self.support] = values
        self.output.set_sample(sample)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"SupportPrior(gen={gen})"
