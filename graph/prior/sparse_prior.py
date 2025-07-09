import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import reduce_precision_to_scalar, sparse_complex_array
from typing import Optional


class SparsePrior(Prior):
    def __init__(
        self,
        rho=0.5,
        shape=(1,),
        dtype=np.complex128,
        damping=0.0,
        precision_mode: Optional[str] = None,
    ):
        """
        Spike-and-slab prior with sparsity level `rho`.

        Args:
            rho (float): Probability of non-zero component.
            damping (float): Damping factor for message updates.
            precision_mode (str or None): "scalar", "array", or None
        """
        self.rho = rho
        self.damping = damping
        self.old_msg = None

        super().__init__(shape=shape, dtype=dtype, precision_mode=precision_mode)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Compute posterior under spike-and-slab model and convert to message.
        Apply damping if previous message exists and damping > 0.
        """
        posterior = self.approximate_posterior(incoming)
        new_msg = posterior / incoming

        if self.old_msg is not None and self.damping > 0:
            new_msg = new_msg.damp_with(self.old_msg, alpha=self.damping)

        self.old_msg = new_msg
        return new_msg

    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Compute approximate posterior using spike-and-slab prior
        with elementwise moment matching.

        Returns:
            UncertainArray: belief-like object representing posterior.
        """
        m = incoming.data
        v = 1.0 / incoming._precision  # Note: precision may be scalar or array

        prec_post = 1.0 + 1.0 / v
        v_post = 1.0 / prec_post
        m_post = v_post * (m / v)

        slab_likelihood = self.rho * np.exp(-np.abs(m) ** 2 / (1.0 + v)) / (1.0 + v)
        spike_likelihood = (1 - self.rho) * np.exp(-np.abs(m) ** 2 / v) / v
        Z = slab_likelihood + spike_likelihood + 1e-12

        mu = (slab_likelihood / Z) * m_post
        e_x2 = (slab_likelihood / Z) * (np.abs(m_post) ** 2 + v_post)
        var = np.maximum(e_x2 - np.abs(mu) ** 2, 1e-12)

        precision = 1.0 / var
        if self.output.precision_mode == "scalar":
            precision = reduce_precision_to_scalar(precision)

        return UA(mu, dtype=self.dtype, precision=precision)

    def generate_sample(self, rng):
        """
        Generate a sparse sample from spike-and-slab prior.
        """
        sample = sparse_complex_array(self.shape, sparsity=self.rho, dtype=self.dtype, rng=rng)
        self.output.set_sample(sample)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"SPrior(gen={gen}, mode={self.precision_mode})"
