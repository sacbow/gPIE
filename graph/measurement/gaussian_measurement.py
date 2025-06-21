import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from graph.wave import Wave

class GaussianMeasurement(Measurement):
    def __init__(self, input_wave: Wave, observed_array, var=1.0, dtype=np.complex128):
        """
        Gaussian measurement model:
        y ~ CN(x, var)

        Args:
            input_wave (Wave): The wave variable being measured.
            observed_array (ndarray): Observed complex array (raw values).
            var (float): Noise variance (default 1.0).
            dtype: Data type (default np.complex128).
        """
        observed = UA(observed_array, dtype=dtype, precision=1.0 / var)
        super().__init__(input_wave, observed)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the observed distribution as the message (fixed Gaussian).
        """
        return self.observed
