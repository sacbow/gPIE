import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA

class SparsePrior(Prior):
    def __init__(self, rho=0.5, shape=(1,), dtype=np.complex128, damping=0.0, seed=None):
        """
        Spike-and-slab prior with sparsity level `rho`.
        Belief is approximated as a Gaussian and optionally damped.
        """
        self.rho = rho
        self.damping = damping
        self.belief = None

        super().__init__(shape=shape, dtype=dtype, seed=seed)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Compute the outgoing message based on approximated posterior.
        Applies optional damping to the belief before generating the message.
        """
        posterior = self.approximate_posterior(incoming)

        if self.belief is not None and self.damping > 0:
            posterior = posterior.damp_with(self.belief, alpha=self.damping)

        self.belief = posterior
        return posterior / incoming

    def forward(self):
        """
        Overrides base forward() to handle first-time initialization of belief.
        """
        if self.output_message is None:
            msg = UA.random(self.shape, dtype=self.dtype, seed = self.seed)
            self.output_message = msg
            self.output.receive_message(self, msg)
            self.belief = UA.zeros(self.shape, dtype=self.dtype, precision=1.0)
            return

        msg = self._compute_message(self.output_message)
        self.output_message = msg
        self.output.receive_message(self, msg)

    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Compute the approximate posterior distribution under spike-and-slab prior.
        This is done via moment matching, applied elementwise.
        """
        m = incoming.data
        v = 1.0 / incoming.precision

        # Posterior of slab: combine CN(0,1) and CN(m,v)
        prec_post = 1.0 + 1.0 / v
        v_post = 1.0 / prec_post
        m_post = v_post * (m / v)

        # Marginal likelihoods
        slab_likelihood = self.rho * np.exp(-np.abs(m)**2 / (1.0 + v)) / (np.pi * (1.0 + v))
        spike_likelihood = (1 - self.rho) * np.exp(-np.abs(m)**2 / v) / (np.pi * v)
        Z = slab_likelihood + spike_likelihood + 1e-12  # avoid divide-by-zero

        # Posterior mean
        mu = (slab_likelihood / Z) * m_post

        # Posterior variance = E[|x|^2] - |E[x]|^2
        e_x2 = (slab_likelihood / Z) * (np.abs(m_post)**2 + v_post)
        var = e_x2 - np.abs(mu)**2
        var = np.maximum(var, 1e-12)  # ensure positive

        precision = 1.0 / var
        return UA(mu, dtype=self.dtype, precision=precision)
