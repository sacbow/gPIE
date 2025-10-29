"""
Adaptive damping scheduler for expectation propagation.

This module implements an AD-GAMP-like adaptive damping rule that adjusts the
damping parameter (or equivalently, step size) based on the recent trajectory of
a any user-defined scalar cost.

The adaptive rule follows the method proposed in:

    J. Vila, P. Schniter, S. Rangan, F. Krzakala and L. Zdeborová,
    "Adaptive damping and mean removal for the generalized approximate message passing algorithm,"
    2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
    South Brisbane, QLD, Australia, 2015, pp. 2021–2025,
    doi: 10.1109/ICASSP.2015.7178325.

In that paper, the adaptive rule monitors a scalar objective J_t and increases or decreases 
the damping factor β ∈ (0,1] depending on whether J_t improves compared to recent iterations. 
This implementation is general and can be plugged into arbitrary factor graph nodes in gPIE.

The mapping between AMP's β and gPIE's `damping` parameter is:
    damping = 1 - β
where gPIE's `damping` = 0 corresponds to "no damping", and `damping` = 1 corresponds
to "fully damped" (frozen) updates.

Typical default parameters:
    G_pass = 1.1, G_fail = 0.5, β_min = 0.01, β_max = 1.0, T_β = 3
"""

from collections import deque
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DampingScheduleConfig:
    """
    Configuration parameters for the adaptive damping scheduler.

    Attributes
    ----------
    G_pass : float
        Multiplicative factor applied to β when the iteration is successful ("pass").
        Must be > 1. Typical value: 1.05–1.2.
    G_fail : float
        Multiplicative factor applied to β when the iteration fails ("fail").
        Must be in (0,1). Typical value: 0.3–0.7.
    beta_min : float
        Lower bound of β (corresponds to maximum damping in gPIE convention).
    beta_max : float
        Upper bound of β (corresponds to no damping in gPIE convention).
    T_beta : int
        Size of the sliding window to compare the current objective value
        against the maximum of the past T_beta iterations.
    """

    G_pass: float = 1.1
    G_fail: float = 0.5
    beta_min: float = 0.01
    beta_max: float = 1.0
    T_beta: int = 3


class AdaptiveDamping:
    """
    Adaptive damping controller for a single node or factor.

    This class implements the iterative rule described in Table I of
    Vila et al. (ICASSP 2015), with a flexible objective function J_t.

    The internal variable β ∈ (0,1] is updated according to:
        if (J_t <= max{J_{t-1}, ..., J_{t-T_β}}) or (β <= β_min):
            β ← min(β_max, G_pass * β)
            pass = True
        else:
            β ← max(β_min, G_fail * β)
            pass = False

    In gPIE, damping = 1 - β, so increasing β reduces damping and vice versa.

    Example
    -------
    >>> sched = AdaptiveDamping(DampingScheduleConfig())
    >>> damping, repeat = sched.step(J=1.23)
    >>> print(damping, repeat)
    0.0 False
    """

    def __init__(self, cfg: DampingScheduleConfig):
        self.cfg = cfg
        self.beta: float = cfg.beta_max  # Start with minimal damping (β=β_max)
        self.hist: deque[float] = deque(maxlen=max(1, cfg.T_beta))

    def step(self, J: float) -> Tuple[float, bool]:
        """
        Update the internal damping state based on the new objective value.

        Parameters
        ----------
        J : float
            The current objective value to evaluate (e.g., negative log-likelihood or local KL).

        Returns
        -------
        damping : float
            The new gPIE damping value in [0, 1], mapped as damping = 1 - β.
        repeat_same_iter : bool
            Whether to repeat the same outer iteration (True if fail condition is triggered).
        """
        # Get the worst (maximum) objective in recent history
        worst_recent = max(self.hist) if self.hist else float("inf")

        # Determine pass/fail condition
        passed = (J <= worst_recent) or (self.beta <= self.cfg.beta_min)

        if passed:
            # Successful iteration: relax damping (increase β)
            self.beta = min(self.cfg.beta_max, self.cfg.G_pass * self.beta)
            self.hist.append(J)
            repeat = False
        else:
            # Failed iteration: tighten damping (decrease β)
            self.beta = max(self.cfg.beta_min, self.cfg.G_fail * self.beta)
            repeat = True

        # Convert to gPIE's convention
        damping = 1.0 - self.beta

        return damping, repeat

    def reset(self):
        """Reset the internal β and history to their initial states."""
        self.beta = self.cfg.beta_max
        self.hist.clear()

    def __repr__(self) -> str:
        return (
            f"AdaptiveDamping(beta={self.beta:.4f}, "
            f"hist_len={len(self.hist)}, cfg={self.cfg})"
        )
