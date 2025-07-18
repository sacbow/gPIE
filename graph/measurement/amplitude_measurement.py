import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from core.types import PrecisionMode
from typing import Optional, Union


class AmplitudeMeasurement(Measurement):
    """
    Measurement model: y = |z| + N(0, var), real-valued observation
    """

    input_dtype: np.dtype = np.complex128
    expected_observed_dtype: np.dtype = np.float64

    def __init__(
        self,
        observed_data: Optional[np.ndarray] = None,
        var: float = 1e-4,
        damping: float = 0.0,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self._var = var
        self.damping = damping
        self.old_msg: Optional[UA] = None

        if precision_mode is None and mask is not None:
            precision_mode = PrecisionMode.ARRAY
        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode)

        self._mask = mask

        if mask is not None and precision_mode == PrecisionMode.SCALAR:
            raise ValueError("Masked observation requires array precision mode, not 'scalar'.")

        # Construct observed UA if observed_data is provided
        observed: Optional[UA] = None
        if observed_data is not None:
            if not np.issubdtype(observed_data.dtype, np.floating):
                raise TypeError("Observed data must be real-valued.")
            if mask is not None:
                if observed_data.shape != mask.shape:
                    raise ValueError("observed_data and mask must match in shape.")
                precision = np.where(mask, 1.0 / var, 0.0)
            elif precision_mode == PrecisionMode.SCALAR or precision_mode is None:
                precision = 1.0 / var
            else:
                precision = np.full_like(observed_data, 1.0 / var, dtype=np.float64)

            observed = UA(observed_data, dtype=self.expected_observed_dtype, precision=precision)

        super().__init__(observed=observed, precision_mode=precision_mode, mask=mask)

    def generate_sample(self, rng: np.random.Generator) -> None:
        """
        Generate noisy amplitude: y = |x| + N(0, sqrt(var))
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        abs_x = np.abs(x)
        noise = rng.normal(loc=0.0, scale=np.sqrt(self._var), size=abs_x.shape)
        self._sample = abs_x + noise

    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Compute approximate posterior using Laplace approximation.

        This implements an amplitude-domain posterior approximation based on 
        Laplace's method, applied elementwise to each input.

        The technique follows the approach described in:

            S. K. Shastri, R. Ahmad, and P. Schniter,
            "Deep Expectation-Consistent Approximation for Phase Retrieval,"
            in *Proc. Asilomar Conf. on Signals, Systems, and Computers*, 2023.
            DOI: 10.1109/IEEECONF59524.2023.10476950

        Returns:
            UncertainArray: Approximate posterior distribution over the input.
        """

        z0 = incoming.data
        tau = incoming.precision(raw=True)
        v0 = 1.0 / tau
        y = self.observed.data
        v = self._var

        abs_z0 = np.abs(z0)
        abs_z0_safe = np.maximum(abs_z0, 1e-12)
        unit_phase = z0 / abs_z0_safe

        z_hat = (v0 * y + 2 * v * abs_z0_safe) / (v0 + 2 * v) * unit_phase
        v_hat = (v0 * (v0 * y + 4 * v * abs_z0_safe)) / (2 * abs_z0_safe * (v0 + 2 * v))
        v_hat = np.maximum(v_hat, 1e-12)

        posterior = UA(z_hat, dtype=self.input_dtype, precision=1.0 / v_hat)

        if self.precision_mode_enum == PrecisionMode.SCALAR:
            return posterior.as_scalar_precision()
        return posterior

    def _compute_message(self, incoming: UA) -> UA:
        """
        Compute backward message based on Laplace posterior.
        Handles masked regions explicitly.
        """
        self._check_observed()
        belief = self.approximate_posterior(incoming)

        if self._mask is not None:
            m = self._mask
            full_msg = belief / incoming
            msg_data = np.zeros_like(full_msg.data)
            msg_prec = np.zeros_like(full_msg.precision(raw=True))
            msg_data[m] = full_msg.data[m]
            msg_prec[m] = full_msg.precision(raw=True)[m]
            msg = UA(msg_data, dtype=self.input_dtype, precision=msg_prec)
        else:
            msg = belief / incoming
            if self.precision_mode_enum == PrecisionMode.SCALAR:
                msg = msg.as_scalar_precision()

        if self.old_msg is not None and self.damping > 0:
            msg = msg.damp_with(self.old_msg, alpha=self.damping)

        self.old_msg = msg
        return msg

    def set_observed(self, data: np.ndarray, var: Optional[float] = None) -> None:
        """
        Manually assign observed data and (optionally) noise variance.
        """
        var = var if var is not None else self._var
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError("Observed data must be real-valued.")

        if self.precision_mode_enum == PrecisionMode.SCALAR:
            precision = 1.0 / var
        else:
            precision = np.full_like(data, fill_value=1.0 / var, dtype=np.float64)

        self.observed = UA(data, dtype=self.expected_observed_dtype, precision=precision)

    def update_observed_from_sample(self) -> None:
        """
        Generate observed UA from current sample.
        """
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        if self.precision_mode_enum == PrecisionMode.SCALAR:
            precision = 1.0 / self._var
        else:
            precision = np.full_like(self._sample, fill_value=1.0 / self._var, dtype=np.float64)

        self.observed = UA(self._sample, dtype=self.expected_observed_dtype, precision=precision)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"AmplitudeMeas(gen={gen}, mode={self.precision_mode})"
