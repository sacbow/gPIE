import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_normal_array
from graph.wave import Wave


class GaussianMeasurement(Measurement):
    input_dtype = np.complex128
    expected_observed_dtype = np.complex128

    def __init__(self, input_wave: Wave, observed_array=None, var=1.0, precision_mode=None):
        """
        Gaussian measurement model: y ~ CN(x, var)
        `observed_array` may be set later via set_observed or update_observed_from_sample.

        Args:
            input_wave (Wave): Connected input wave.
            observed_array (ndarray): Optional observed data array.
            var (float): Observation noise variance.
            precision_mode (str or None): "scalar" or "array", or None for inference.
        """
        self._var = var
        self._precision_value = 1.0 / var
        self.precision_mode = precision_mode

        if observed_array is not None:
            precision = (
                self._precision_value
                if precision_mode == "scalar"
                else np.ones_like(observed_array, dtype=np.float64) * self._precision_value
            )
            observed = UA(observed_array, dtype=self.expected_observed_dtype, precision=precision)
        else:
            observed = None

        super().__init__(input_wave=input_wave, observed=observed)

    def generate_sample(self, rng):
        """
        Generate noisy observation y = x + noise, and store it in self._sample.
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        noise = random_normal_array(x.shape, dtype=self.input_dtype, rng=rng)
        y = x + noise * np.sqrt(self._var)
        self._sample = y

    def set_observed(self, data):
        """
        Set observed data explicitly. Precision shape depends on self.precision_mode.
        """
        if self.precision_mode == "scalar":
            precision = self._precision_value
        else:
            precision = np.ones_like(data, dtype=np.float64) * self._precision_value

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