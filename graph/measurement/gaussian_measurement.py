import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_normal_array
from graph.wave import Wave


class GaussianMeasurement(Measurement):
    input_dtype = np.complex128
    expected_observed_dtype = np.complex128

    def __init__(self, input_wave: Wave, observed_array=None, var=1.0):
        """
        Gaussian measurement model: y ~ CN(x, var)
        `observed_array` may be set later via set_observed or update_observed_from_sample.
        """
        self._var = var

        if observed_array is not None:
            precision = np.ones_like(observed_array, dtype=np.float64) / var
            observed = UA(observed_array, dtype=self.expected_observed_dtype, precision=precision)
        else:
            observed = None

        super().__init__(input_wave=input_wave, observed=observed)

    def generate_sample(self, rng):
        """
        Generate noisy observation y = x + noise, and store it in self._sample.
        Does not automatically update self.observed.
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        noise = random_normal_array(x.shape, dtype=self.input_dtype, rng=rng)
        y = x + noise * np.sqrt(self._var)
        self._sample = y

    def set_observed(self, data):
        """
        Set observed data explicitly.
        """
        precision = np.ones_like(data, dtype=np.float64) / self._var
        super().set_observed(data, precision=precision, dtype=self.expected_observed_dtype)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the observed distribution as the message (fixed Gaussian).
        """
        self._check_observed()
        return self.observed

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen})"
