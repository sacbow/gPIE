import numpy as np
from .base import Measurement
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_normal_array
from core.types import PrecisionMode
from typing import Optional, Union


class GaussianMeasurement(Measurement):
    """
    Gaussian measurement model: y ~ N(x, var) or CN(x, var).
    """

    input_dtype: Optional[np.dtype] = None
    expected_observed_dtype: Optional[np.dtype] = None  # will default to input_dtype

    def __init__(
        self,
        observed_array: Optional[np.ndarray] = None,
        var: float = 1.0,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        mask: Optional[np.ndarray] = None,
        dtype: Optional[np.dtype] = None,  # input_dtype を明示指定する場合
    ) -> None:
        """
        Initialize Gaussian measurement model.

        Args:
            observed_array: Optional observed data array.
            var: Observation noise variance.
            precision_mode: "scalar", "array", or None.
            mask: Optional observation mask.
            dtype: Optional input dtype (e.g., np.float64 or np.complex128).
        """
        self._var = var
        self._precision_value = 1.0 / var

        if dtype is not None:
            self.input_dtype = dtype
            self.expected_observed_dtype = dtype

        if precision_mode is not None and isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode)

        if mask is not None and precision_mode == PrecisionMode.SCALAR:
            raise ValueError("Masked observation requires array precision mode.")

        observed: Optional[UA] = None
        if observed_array is not None:
            if mask is not None:
                if observed_array.shape != mask.shape:
                    raise ValueError("observed_array and mask must have the same shape.")
                precision = np.where(mask, self._precision_value, 0.0)
            elif precision_mode == PrecisionMode.SCALAR or precision_mode is None:
                precision = self._precision_value
            else:
                precision = np.full_like(observed_array, self._precision_value, dtype=np.float64)

            observed = UA(observed_array, dtype=dtype or np.complex128, precision=precision)

        super().__init__(
            observed=observed,
            precision_mode=precision_mode,
            mask=mask
        )

    def generate_sample(self, rng: np.random.Generator) -> None:
        """
        Generate noisy observation: y = x + noise.
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        noise = random_normal_array(x.shape, dtype=self.input_dtype, rng=rng)
        self._sample = x + np.sqrt(self._var) * noise

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the observed message directly.
        """
        self._check_observed()
        if self.precision_mode_enum == PrecisionMode.SCALAR:
            return self.observed.as_scalar_precision()
        else:
            return self.observed

    def set_observed(self, data: np.ndarray, var: Optional[float] = None) -> None:
        """
        Manually assign observed data and its precision to this measurement.
        """
        var = var if var is not None else self._var
        prec = 1.0 / var

        if self._mask is not None:
            if data.shape != self._mask.shape:
                raise ValueError("Observed data and mask shape mismatch.")
            precision = np.where(self._mask, prec, 0.0)
        elif self.precision_mode_enum == PrecisionMode.SCALAR:
            precision = prec
        else:
            precision = np.full_like(data, fill_value=prec, dtype=np.float64)

        self.observed = UA(data, dtype=self.expected_observed_dtype, precision=precision)


    def update_observed_from_sample(self) -> None:
        """
        Use current sample as observed value, constructing proper masked precision.
        """
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        prec = 1.0 / self._var

        if self._mask is not None:
            if self._sample.shape != self._mask.shape:
                raise ValueError("Sample and mask shape mismatch.")
            precision = np.where(self._mask, prec, 0.0)
        elif self.precision_mode_enum == PrecisionMode.SCALAR:
            precision = prec
        else:
            precision = np.full_like(self._sample, fill_value=prec, dtype=np.float64)

        self.observed = UA(self._sample, dtype=self.expected_observed_dtype, precision=precision)


    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen}, mode={self.precision_mode})"
