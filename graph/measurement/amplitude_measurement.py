import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_normal_array
from graph.wave import Wave


class AmplitudeMeasurement(Measurement):
    input_dtype = np.complex128
    expected_observed_dtype = np.float64

    def __init__(self, input_wave: Wave, observed_data=None, var=1e-4, damping=0.0, precision_mode=None):
        """
        Measurement model: y = |z| + N(0, var), real-valued observation
        """
        self._var = var
        self.damping = damping
        self.old_msg = None
        self.precision_mode = precision_mode

        if observed_data is not None:
            if not np.issubdtype(observed_data.dtype, np.floating):
                raise TypeError("observed_data must be real-valued.")
            if precision_mode == "scalar":
                precision = 1.0 / var
            else:
                precision = np.full_like(observed_data, fill_value=1.0 / var, dtype=np.float64)
            observed = UA(observed_data, dtype=self.expected_observed_dtype, precision=precision)
        else:
            observed = None

        super().__init__(input_wave=input_wave, observed=observed)

    def generate_sample(self, rng):
        """
        Generate noisy amplitude: y = |x| + N(0, sqrt(var))
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        abs_x = np.abs(x)
        noise = rng.normal(loc=0.0, scale=np.sqrt(self._var), size=abs_x.shape)
        y = abs_x + noise

        if self.precision_mode == "scalar":
            precision = 1.0 / self._var
        else:
            precision = np.full_like(y, fill_value=1.0 / self._var, dtype=np.float64)

        self._sample = y
        self.observed = UA(y, dtype=self.expected_observed_dtype, precision=precision)
        self.observed_dtype = self.expected_observed_dtype

    def approximate_posterior(self, incoming: UA) -> UA:
        """
        Compute approximate posterior using Laplace approximation
        """
        z0 = incoming.data
        tau = incoming.precision
        v0 = 1.0 / tau
        y = self.observed.data
        v = self._var

        abs_z0 = np.abs(z0)
        abs_z0_safe = np.maximum(abs_z0, 1e-12)
        unit_phase = z0 / abs_z0_safe

        z_hat = (v0 * y + 2 * v * abs_z0_safe) / (v0 + 2 * v) * unit_phase

        v_hat = (v0 * (v0 * y + 4 * v * abs_z0_safe)) / (2 * abs_z0_safe * (v0 + 2 * v))
        v_hat = np.maximum(v_hat, 1e-12)

        return UA(z_hat, dtype=self.input_dtype, precision=1.0 / v_hat)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Compute backward message from Laplace posterior.
        """
        self._check_observed()
        belief = self.approximate_posterior(incoming)

        if self.precision_mode == "scalar":
            belief = belief.as_scalar_precision()

        msg = belief / incoming

        if self.old_msg is not None and self.damping > 0:
            msg = msg.damp_with(self.old_msg, alpha=self.damping)

        self.old_msg = msg
        return msg

    def set_observed(self, data, var=None):
        """
        Manually set observed data with optional noise variance.
        """
        var = var if var is not None else self._var
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError("Observed data must be real-valued.")

        if self.precision_mode == "scalar":
            precision = 1.0 / var
        else:
            precision = np.full_like(data, fill_value=1.0 / var, dtype=np.float64)

        self.observed = UA(data, dtype=self.expected_observed_dtype, precision=precision)

    def update_observed_from_sample(self):
        """
        Regenerate observed UA from current sample
        """
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        if self.precision_mode == "scalar":
            precision = 1.0 / self._var
        else:
            precision = np.full_like(self._sample, fill_value=1.0 / self._var, dtype=np.float64)

        self.observed = UA(self._sample, dtype=self.expected_observed_dtype, precision=precision)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"AmplitudeMeas(gen={gen})"
