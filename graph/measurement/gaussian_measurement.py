import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from graph.wave import Wave


class GaussianMeasurement(Measurement):
    def __init__(self, input_wave: Wave, observed_array=None, var=1.0, dtype=np.complex128):
        """
        Gaussian measurement model: y ~ CN(x, var)
        observed_array may be set later via set_observed or update_observed_from_sample.
        """
        if observed_array is not None:
            precision = np.ones_like(observed_array, dtype=np.float64) / var
            observed = UA(observed_array, dtype=dtype, precision=precision)
        else:
            observed = None

        super().__init__(input_wave, observed)

        self._var = var
        self._dtype = dtype

    def generate_sample(self, rng):
        """
        Generate noisy observation y = x + noise, and store it in self._sample.
        Does not automatically update self.observed.
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        noise = (
            rng.normal(0.0, np.sqrt(self._var / 2), size=x.shape) +
            1j * rng.normal(0.0, np.sqrt(self._var / 2), size=x.shape)
        )
        y = x + noise
        self._sample = y  # Store for later use

    def set_observed(self, data):
        """
        Set observed data explicitly. If var is None, use self._var.
        """
        self.observed = UA(data, dtype=self._dtype, precision=1.0 / self._var)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the observed distribution as the message (fixed Gaussian).
        """
        self._check_observed()
        return self.observed

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen})"
