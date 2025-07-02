import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_normal_array
from graph.wave import Wave


class MaskedGaussianMeasurement(Measurement):
    input_dtype = np.complex128
    expected_observed_dtype = np.complex128

    def __init__(self, input_wave: Wave, observed_data=None, var=1.0, mask=None):
        """
        Gaussian measurement with missing data (masked observation).

        Args:
            input_wave (Wave): Observed wave variable.
            observed_data (ndarray or None): Complex-valued measurement data.
            var (float): Noise variance for observed entries.
            mask (ndarray of bool): True where observation is available.
        """
        if mask is None:
            raise ValueError("mask must be provided.")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("mask must be of boolean dtype.")

        self._var = var
        self._mask = mask

        if observed_data is not None:
            if observed_data.shape != mask.shape:
                raise ValueError("observed_data and mask must have the same shape.")
            precision = np.where(mask, 1.0 / var, 0.0)
            observed = UA(observed_data, dtype=self.expected_observed_dtype, precision=precision)
        else:
            observed = None

        super().__init__(input_wave=input_wave, observed=observed)

    def generate_sample(self, rng):
        """
        Generate masked noisy observation y = x + noise.
        Noise is added only at observed positions.
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        noise = np.zeros_like(x)
        masked_noise = random_normal_array(
            shape=(np.sum(self._mask),),
            dtype=self.input_dtype,
            rng=rng
        )
        noise[self._mask] = masked_noise * np.sqrt(self._var)
        self._sample = x + noise

    def update_observed_from_sample(self):
        """
        Reflect self._sample into self.observed using stored mask/var.
        """
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")
        precision = np.where(self._mask, 1.0 / self._var, 0.0)
        self.observed = UA(self._sample, dtype=self.expected_observed_dtype, precision=precision)

    def set_observed(self, data, var=None, mask=None):
        """
        Set observed data manually with mask and optional noise variance.
        If var or mask is not provided, uses stored self._var and self._mask.
        """
        var = var if var is not None else self._var
        mask = mask if mask is not None else self._mask

        if data.shape != mask.shape:
            raise ValueError("Shape mismatch between data and mask.")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("Mask must be of boolean dtype.")

        precision = np.where(mask, 1.0 / var, 0.0)
        self.observed = UA(data, dtype=self.expected_observed_dtype, precision=precision)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the masked observation directly as message.
        """
        self._check_observed()
        return self.observed

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"MaskedMeas(gen={gen})"
