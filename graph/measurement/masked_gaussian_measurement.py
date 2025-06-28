from .base import Measurement
from core.uncertain_array import UncertainArray as UA
import numpy as np

class MaskedGaussianMeasurement(Measurement):
    def __init__(self, input_wave, observed_data=None, var=1.0, mask=None, dtype=np.complex128):
        """
        Gaussian measurement with missing data (masked observation).

        Args:
            input_wave (Wave): Observed wave variable.
            observed_data (ndarray or None): Complex-valued measurement data.
            var (float): Noise variance for observed entries.
            mask (ndarray of bool): True where observation is available.
            dtype (np.dtype): Data type (default: complex128).
        """
        if mask is None:
            raise ValueError("mask must be provided.")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("mask must be of boolean dtype.")

        if observed_data is not None:
            if observed_data.shape != mask.shape:
                raise ValueError("observed_data and mask must have the same shape.")
            precision = np.where(mask, 1.0 / var, 0.0)
            observed = UA(observed_data, dtype=dtype, precision=precision)
        else:
            observed = None

        super().__init__(input_wave, observed)

        self._mask = mask
        self._var = var
        self._dtype = dtype

    def generate_sample(self, rng):
        """
        Generate masked noisy observation y = x + noise.
        Noise is added only at observed positions.
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        noise = np.zeros_like(x)
        noise[self._mask] = (
            rng.normal(0.0, np.sqrt(self._var / 2), size=np.sum(self._mask)) +
            1j * rng.normal(0.0, np.sqrt(self._var / 2), size=np.sum(self._mask))
        )

        y = x + noise
        self._sample = y

    def update_observed_from_sample(self):
        """
        Reflect self._sample into self.observed using stored mask/var.
        """
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")
        precision = np.where(self._mask, 1.0 / self._var, 0.0)
        self.observed = UA(self._sample, dtype=self._dtype, precision=precision)

    def set_observed(self, data, var, mask):
        """
        Set observed data manually with mask and noise variance.
        """
        if data.shape != mask.shape:
            raise ValueError("Shape mismatch between data and mask.")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("Mask must be of boolean dtype.")

        precision = np.where(mask, 1.0 / var, 0.0)
        self.observed = UA(data, dtype=self._dtype, precision=precision)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the masked observation directly as message.
        """
        return self.observed

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"MaskedMeas(gen={gen})"