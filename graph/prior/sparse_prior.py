import numpy as np
from .base import Prior
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import reduce_precision_to_scalar

class SparsePrior(Prior):
    def __init__(self, rho=0.5, shape=(1,), dtype=np.complex128, damping=0.0, seed=None):
        """
        Spike-and-slab prior with sparsity level `rho`.
        Message damping is applied to outgoing messages, not beliefs.
        """
        self.rho = rho
        self.damping = damping
        self.old_msg = None  # Keep track of last message sent
        super().__init__(shape=shape, dtype=dtype, seed=seed)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Compute posterior under spike-and-slab model, convert to message,
        apply damping against previous message if needed.
        """
        posterior = self.approximate_posterior(incoming)
        new_msg = posterior / incoming

        if self.old_msg is not None and self.damping > 0:
            msg_to_send = new_msg.damp_with(self.old_msg, alpha=self.damping)
        else:
            msg_to_send = new_msg

        self.old_msg = msg_to_send
        return msg_to_send

    def forward(self):
        """
        Send message to output wave, possibly using previous message.
        """
        if self.output_message is None:
            msg = UA.random(self.shape, dtype=self.dtype, seed=self.seed)
            self.old_msg = msg
        else:
            msg = self._compute_message(self.output_message)

        self.output.receive_message(self, msg)

    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Compute approximate posterior using spike-and-slab prior.
        This uses elementwise moment matching.
        """
        m = incoming.data
        v = 1.0 / incoming._precision

        prec_post = 1.0 + 1.0 / v
        v_post = 1.0 / prec_post
        m_post = v_post * (m / v)

        slab_likelihood = self.rho * np.exp(-np.abs(m)**2 / (1.0 + v)) / (1.0 + v)
        spike_likelihood = (1 - self.rho) * np.exp(-np.abs(m)**2 / v) / v
        Z = slab_likelihood + spike_likelihood + 1e-12  # avoid divide-by-zero

        mu = (slab_likelihood / Z) * m_post
        e_x2 = (slab_likelihood / Z) * (np.abs(m_post)**2 + v_post)
        var = e_x2 - np.abs(mu)**2
        var = np.maximum(var, 1e-12)

        precision = 1.0 / var
        return UA(mu, dtype=self.dtype, precision=reduce_precision_to_scalar(precision))
