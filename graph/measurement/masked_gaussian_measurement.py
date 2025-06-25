from .base import Measurement
from core.uncertain_array import UncertainArray as UA
import numpy as np

class MaskedGaussianMeasurement(Measurement):
    def __init__(self, input_wave, observed_data, var, mask, dtype=np.complex128):
        """
        Gaussian measurement with missing data (masked observation).

        Args:
            input_wave (Wave): Observed wave variable.
            observed_data (ndarray): Complex-valued measurement data.
            var (float): Noise variance for observed entries.
            mask (ndarray of bool): True where observation is available.
            dtype (np.dtype): Data type (default: complex128).
        """
        if observed_data.shape != mask.shape:
            raise ValueError("observed_data and mask must have the same shape.")
        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError("mask must be of boolean dtype.")

        precision = np.where(mask, 1.0 / var, 0.0)
        observed = UA(observed_data, dtype=dtype, precision=precision)

        super().__init__(input_wave, observed)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the masked observation directly as message.

        The observation has precision=1/var at observed entries and 0 elsewhere.
        This message expresses the likelihood function CN(y | x, var) (masked).
        """
        return self.observed


