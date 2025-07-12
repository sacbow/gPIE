import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_normal_array
from graph.wave import Wave


class GaussianMeasurement(Measurement):
    input_dtype = np.complex128
    expected_observed_dtype = np.complex128

    def __init__(self,
                 observed_array=None,
                 var=1.0,
                 precision_mode=None,
                 mask=None):
        """
        Gaussian measurement model: y ~ CN(x, var)

        Args:
            input_wave (Wave): Connected input wave.
            observed_array (ndarray or None): Optional observed data.
            var (float): Observation noise variance.
            precision_mode (str or None): "scalar", "array", or None.
            mask (ndarray of bool or None): Optional mask indicating valid observations.
        """
        self._var = var
        self._precision_value = 1.0 / var

        if observed_array is not None:
            if mask is not None:
                if observed_array.shape != mask.shape:
                    raise ValueError("observed_array and mask must have the same shape.")
                precision = np.where(mask, self._precision_value, 0.0)
            elif precision_mode == "scalar" or precision_mode is None:
                # 明示的に scalar またはデフォルト（通常はこちら）
                precision = self._precision_value
            else:
                precision = np.full_like(observed_array, self._precision_value, dtype=np.float64)

            observed = UA(observed_array, dtype=self.expected_observed_dtype, precision=precision)
        else:
            observed = None

        super().__init__(
            observed=observed,
            precision_mode=precision_mode,
            mask=mask
        )

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

    def _compute_message(self, incoming: UA) -> UA:
        self._check_observed()
        if self.precision_mode == "scalar":
            return self.observed.as_scalar_precision()
        else:
            return self.observed


    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen})"
