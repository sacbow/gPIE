from typing import Optional, Union, Any

from .base import Measurement
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_real_dtype
from ...core.rng_utils import normal


class AmplitudeMeasurement(Measurement):
    """
    Nonlinear amplitude measurement model: y = |z| + ε, with ε ~ N(0, var)

    Observes the magnitude of a complex-valued latent variable plus additive Gaussian noise.
    """

    expected_input_dtype = np().complexfloating
    expected_observed_dtype = np().floating

    def __init__(
        self,
        var: float = 1e-4,
        damping: float = 0.0,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        with_mask: bool = False
    ) -> None:
        self._var = var
        self.damping = damping
        self.old_msg: Optional[UA] = None

        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode)

        super().__init__(with_mask=with_mask)
        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

    def _infer_observed_dtype_from_input(self, input_dtype: np().dtype) -> np().dtype:
        return get_real_dtype(input_dtype)

    def _generate_sample(self, rng: Any) -> None:
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        abs_x = np().abs(x)
        noise = normal(std=self._var ** 0.5, size=abs_x.shape, rng=rng)
        self._sample = (abs_x + noise).astype(self.observed_dtype)


    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Compute approximate posterior using Laplace approximation.

        This method implements a nonlinear belief update for amplitude measurements
        of the form `y = |z| + ε`, where `z` is a complex latent variable and
        `ε ~ N(0, var)` is additive Gaussian noise.

        The update is based on a Laplace approximation to the posterior distribution,
        computed elementwise using a closed-form expression derived from second-order
        expansion of the likelihood.

        Reference:
            S. K. Shastri, R. Ahmad, and P. Schniter,
            "Deep Expectation-Consistent Approximation for Phase Retrieval,"
            Proc. Asilomar Conf. on Signals, Systems, and Computers, 2023.
            DOI: 10.1109/IEEECONF59524.2023.10476950

        Args:
            incoming (UncertainArray):
                Current belief (mean and precision) on the latent complex variable `z`.

        Returns:
            UncertainArray:
                Posterior approximation as a complex Gaussian belief, in batch form.
        """
        z0 = incoming.data
        tau = incoming.precision(raw=True)
        v0 = np().reciprocal(tau)
        y = self.observed.data
        v = np().reciprocal(self.observed.precision(raw=True))
        eps = np().array(1e-8, dtype=v0.dtype)

        abs_z0 = np().abs(z0)
        abs_z0_safe = np().maximum(abs_z0, eps)
        unit_phase = z0 / abs_z0_safe

        z_hat = (v0 * y + 2 * v * abs_z0_safe) / (v0 + 2 * v) * unit_phase
        v_hat = (v0 * (v0 * y + 4 * v * abs_z0_safe)) / (2 * abs_z0_safe * (v0 + 2 * v))
        v_hat = np().maximum(v_hat, eps)

        posterior = UA(z_hat, dtype=self.input_dtype, precision= np().reciprocal(v_hat))

        if self.precision_mode_enum == PrecisionMode.SCALAR:
            return posterior.as_scalar_precision()
        return posterior

    def _compute_message(self, incoming: UA) -> UA:
        self._check_observed()
        belief = self.approximate_posterior(incoming)
        full_msg = belief / incoming

        if self._mask is not None:
            m = self._mask
            msg_data = np().zeros_like(full_msg.data)
            msg_prec = np().zeros_like(full_msg.precision(raw=True))
            msg_data[m] = full_msg.data[m]
            msg_prec[m] = full_msg.precision(raw=True)[m]
            msg = UA(msg_data, dtype=self.input_dtype, precision=msg_prec)
        else:
            msg = full_msg
            if self.precision_mode_enum == PrecisionMode.SCALAR:
                msg = msg.as_scalar_precision()

        if self.old_msg is not None and self.damping > 0:
            msg = msg.damp_with(self.old_msg, alpha=self.damping)

        self.old_msg = msg
        return msg

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"AmplitudeMeas(gen={gen}, mode={self.precision_mode})"
