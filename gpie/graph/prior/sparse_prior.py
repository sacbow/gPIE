from ...core.backend import np
from ...core.rng_utils import get_rng
from typing import Optional, Any, Union

from .base import Prior
from ...core.adaptive_damping import AdaptiveDamping, DampingScheduleConfig 
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import reduce_precision_to_scalar, random_normal_array
from ...core.types import PrecisionMode, get_real_dtype


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
        event_shape: tuple[int, ...] = (1,),
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        damping: Union[float, str] = "auto",
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None,
        adaptive_cfg: Optional[DampingScheduleConfig] = None
    ) -> None:
        real_dtype = get_real_dtype(dtype)
        self.rho = real_dtype(rho)
        self.old_msg: Optional[UA] = None

        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label,
        )

        if damping == "auto":
            self._adaptive = True
            self._scheduler = AdaptiveDamping(adaptive_cfg or DampingScheduleConfig())
            self.damping = 1.0 - self._scheduler.beta
        else:
            self._adaptive = False
            self.damping = real_dtype(damping)

    def _compute_message(self, incoming: UA) -> UA:
        posterior = self.approximate_posterior(incoming)
        new_msg = posterior / incoming

        if self.old_msg is not None and self.damping > 0:
            new_msg = new_msg.damp_with(self.old_msg, alpha=self.damping)

        self.old_msg = new_msg
        return new_msg

    def approximate_posterior(self, incoming: UA) -> UA:
        m = incoming.data
        v = 1 / incoming.precision(raw=True)
        prec_post = 1 + 1 / v
        v_post = 1 / prec_post
        m_post = v_post * (m / v)
        is_real = incoming.is_real()
        eps = np().array(1e-12, dtype=v.dtype)

        if is_real:
            slab = self.rho * (1 / np().sqrt(1 + v)) * np().exp(- m**2 /(2*(1 + v)))
            spike = (1 - self.rho) * (1 / np().sqrt(v)) * np().exp(-m**2 / (2*v))
        else:
            slab = self.rho * np().exp(-np().abs(m)**2 / (1 + v)) / (1 + v)
            spike = (1 - self.rho) * np().exp(-np().abs(m)**2 / v) / v

        Z = slab + spike + eps  # prevent division by zero

        # --- Store scalar log-likelihood proxy ---
        self.logZ = float(np().sum(np().log(Z + eps)))

        mu = (slab / Z) * m_post
        e_x2 = (slab / Z) * (np().abs(m_post)**2 + v_post)
        var = np().maximum(e_x2 - np().abs(mu)**2, eps)
        precision = 1 / var

        if self.output.precision_mode_enum == PrecisionMode.SCALAR:
            precision = reduce_precision_to_scalar(precision)

        return UA(mu, dtype=self.dtype, precision=precision)

    def forward(self) -> None:
        """
        Send the forward message to the output wave, with optional adaptive damping.
        """
        # --- Standard Prior forward behavior ---
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured for Prior. "
                                "Call graph.set_init_rng(...) before run().")
            msg = self._get_initial_message(self._init_rng)
        else:
            msg = self._compute_message(self.output_message)

        self.output.receive_message(self, msg)

        # --- Adaptive damping control ---
        if self._adaptive is True:
            try:
                J = -self.logZ   # smaller logZ → worse consistency
                new_damp, repeat = self._scheduler.step(J)
                self.damping = new_damp
            except Exception:
                pass



    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        if rng is None:
            rng = get_rng()

        shape = (self.batch_size,) + self.event_shape

        mask = rng.uniform(size=shape) < self.rho
        sample = np().zeros(shape, dtype=self.dtype)
        values = random_normal_array(shape, dtype=self.dtype, rng=rng)
        sample[mask] = values[mask]
        return sample



    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"SPrior(gen={gen}, mode={mode}, rho={self.rho})"
