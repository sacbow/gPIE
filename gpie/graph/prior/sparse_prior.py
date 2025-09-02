from ...core.backend import np
from ...core.rng_utils import get_rng
from typing import Optional, Any

from .base import Prior
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import reduce_precision_to_scalar, random_normal_array
from ...core.types import PrecisionMode


class SparsePrior(Prior):
    """
    A spike-and-slab prior promoting sparsity in the latent variable.

    This prior models each element as:
        x_i ~ (1 - rho) * delta(0) + rho * CN(0, 1)   (or N(0,1) for real dtype)

    During inference, it approximates the posterior via elementwise Bayesian
    updates and maintains damping across iterations to stabilize convergence.

    Behavior:
        - The prior computes an approximate posterior based on the incoming message
        - This posterior is fused with the incoming message to yield an updated message
        - Optional damping is supported to smooth updates across iterations

    Key Components:
        - rho: Probability of the non-zero component (sparsity level)
        - `approximate_posterior()`: Performs spike-and-slab posterior approximation
        - `damping`: Convex weight between previous and new message for stability

    Sampling:
        - Each element is drawn from CN(0,1) (or N(0,1)) with probability rho, or zero otherwise

    Args:
        rho (float): Probability of non-zero entry (default: 0.5).
        shape (tuple[int, ...]): Shape of the latent variable.
        dtype (np().dtype): np().float64 or np().complex128.
        damping (float): Damping coefficient in [0, 1] for belief updates.
        precision_mode (PrecisionMode | None): Scalar or array precision model.
        label (str | None): Optional label for the Wave.

    Attributes:
        rho (float): Non-zero entry probability.
        damping (float): Damping factor.
        old_msg (UncertainArray | None): Previous message (used if damping > 0).
    """

    def __init__(
        self,
        rho: float = 0.5,
        shape: tuple[int, ...] = (1,),
        dtype: np().dtype = np().complex128,
        damping: float = 0.0,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None
    ) -> None:

        self.rho = rho
        self.damping = damping
        self.old_msg: Optional[UA] = None

        super().__init__(shape=shape, dtype=dtype, precision_mode=precision_mode, label=label)

    def _compute_message(self, incoming: UA) -> UA:
        posterior = self.approximate_posterior(incoming)
        new_msg = posterior / incoming

        if self.old_msg is not None and self.damping > 0:
            new_msg = new_msg.damp_with(self.old_msg, alpha=self.damping)

        self.old_msg = new_msg
        return new_msg

    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Approximate posterior using elementwise spike-and-slab.
        Real and complex Gaussian handled separately.
        """
        m = incoming.data
        v = 1.0 / incoming.precision(raw=True)

        prec_post = 1.0 + 1.0 / v
        v_post = 1.0 / prec_post
        m_post = v_post * (m / v)

        is_real = incoming.is_real()

        if is_real:
            # Real-valued Gaussian density
            slab = self.rho * (1.0 / np().sqrt(1 + v)) * np().exp(-0.5 * m**2 / (1 + v))
            spike = (1 - self.rho) * (1.0 / np().sqrt(v)) * np().exp(-0.5 * m**2 / v)
        else:
            # Complex-valued Gaussian density
            slab = self.rho * np().exp(-np().abs(m)**2 / (1 + v)) / ((1 + v))
            spike = (1 - self.rho) * np().exp(-np().abs(m)**2 / v) / v

        Z = slab + spike + 1e-8  # normalization constant

        mu = (slab / Z) * m_post
        e_x2 = (slab / Z) * (np().abs(m_post) ** 2 + v_post)
        var = np().maximum(e_x2 - np().abs(mu) ** 2, 1e-8)
        precision = 1.0 / var

        if self.output.precision_mode_enum == PrecisionMode.SCALAR:
            precision = reduce_precision_to_scalar(precision)

        return UA(mu, dtype=self.dtype, precision=precision)

    def generate_sample(self, rng: Optional[Any]) -> None:
        """
        Generate sparse sample: zero with 1-rho prob, N(0,1) or CN(0,1) with rho.
        """
        if rng is None:
            rng = get_rng()

        mask = rng.uniform(size=self.shape) < self.rho
        sample = np().zeros(self.shape, dtype=self.dtype)
        values = random_normal_array(self.shape, dtype=self.dtype, rng=rng)
        sample[mask] = values[mask]
        self.output.set_sample(sample)

    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return a sparse sample: zero with 1-rho prob, N(0,1) or CN(0,1) with rho.

        Args:
            rng (Optional[Any]): Optional random generator.

        Returns:
            np().ndarray: Sample array with the same shape and dtype as the prior.
        """
        if rng is None:
            rng = get_rng()

        mask = rng.uniform(size=self.shape) < self.rho
        sample = np().zeros(self.shape, dtype=self.dtype)
        values = random_normal_array(self.shape, dtype=self.dtype, rng=rng)
        sample[mask] = values[mask]
        return sample


    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"SPrior(gen={gen}, mode={mode}, rho={self.rho})"
