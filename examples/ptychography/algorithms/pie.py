"""
Standard PIE algorithm (Rodenburg & Faulkner, 2004).

Implements the classic object update rule described in:
J. M. Rodenburg and H. M. L. Faulkner,
"A phase retrieval algorithm for shifting illumination,"
Appl. Phys. Lett. 85, 4795–4797 (2004).
See also A. Maiden et al., Optica 4, 736 (2017).
"""

from gpie.core.backend import np
from .base_pie import BasePIE


class PIE(BasePIE):
    """
    Standard Ptychographical Iterative Engine (PIE).

    The PIE algorithm updates the object by a weighted correction
    derived from the probe and the difference between the projected
    and estimated exit waves:

        O'(r) = O(r) + [P*(r) / (|P(r)|^2 + α |P|_max^2)] * (ψ'(r) - ψ(r))

    where α is a small positive parameter (typically 0.01–0.1) that
    regularizes updates in low-probe-intensity regions.

    Attributes
    ----------
    alpha : float
        Regularization factor controlling the update strength.
    """

    def _update_object(self, proj_wave, exit_wave, indices):
        xp = self.xp
        yy, xx = indices
        diff = proj_wave - exit_wave

        # Probe intensity normalization
        prb_amp2 = xp.abs(self.prb) ** 2
        prb_amp_max2 = xp.max(prb_amp2)

        # Denominator with regularization
        denom = prb_amp2 + self.alpha * prb_amp_max2

        # Update rule
        update = (self.prb.conj() * diff) / denom
        self.obj[yy, xx] += update
