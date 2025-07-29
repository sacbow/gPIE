from ...core.backend import np
from .base import Measurement
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import random_normal_array
from ...core.types import PrecisionMode, get_real_dtype
from typing import Optional, Union, Any


class GaussianMeasurement(Measurement):
    """
    Gaussian measurement model: y ~ N(x, var) or CN(x, var).

    This class implements a measurement factor where the observed data `y`
    is a noisy version of the latent variable `x`, with additive Gaussian noise:

        y = x + ε,     ε ~ N(0, var)   (real or complex)

    This is the most standard observation model used in signal processing
    and probabilistic inference.

    Features:
        - Supports both real and complex data types
        - Accepts scalar or per-element (array) precision
        - Supports optional observation masking (via binary mask)
        - Allows both fixed and synthetic (generated) observations

    Usage:
        meas = GaussianMeasurement(observed_array=..., var=0.1) @ x

    Attributes:
        _var (float): Observation noise variance
        _precision_value (float): 1 / _var
        input_dtype (np().dtype): Required dtype for latent input wave
        expected_observed_dtype (np().dtype): Must match observed data
        observed (UncertainArray): Actual observed values with precision
    """


    input_dtype: Optional[np().dtype] = None
    expected_observed_dtype: Optional[np().dtype] = None  # will default to input_dtype

    def __init__(
        self,
        observed_array: Optional[np().ndarray] = None,
        var: float = 1.0,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        mask: Optional[np().ndarray] = None,
        dtype: Optional[np().dtype] = None,  # input_dtype を明示指定する場合
    ) -> None:
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
            if dtype is not None:
                if np().issubdtype(dtype, np().floating) and not np().issubdtype(observed_array.dtype, np().floating):
                    raise TypeError(
                        f"Observed array dtype {observed_array.dtype} is incompatible with specified real dtype {dtype}"
                    )
                if np().issubdtype(dtype, np().complexfloating) and not np().issubdtype(observed_array.dtype, np().complexfloating):
                    raise TypeError(
                        f"Observed array dtype {observed_array.dtype} is incompatible with specified complex dtype {dtype}"
                    )
            if self.expected_observed_dtype is None:
                self.expected_observed_dtype = observed_array.dtype

            if mask is not None:
                if observed_array.shape != mask.shape:
                    raise ValueError("observed_array and mask must have the same shape.")
                precision = np().where(mask, self._precision_value, 0.0)
            elif precision_mode == PrecisionMode.SCALAR or precision_mode is None:
                precision = self._precision_value
            else:
                precision = np().full_like(
                    observed_array,
                    self._precision_value,
                    dtype=get_real_dtype(observed_array.dtype)
                )

            auto_dtype = dtype or getattr(observed_array, "dtype", None)
            observed = UA(observed_array, dtype=auto_dtype, precision=precision)

        super().__init__(
            observed=observed,
            precision_mode=precision_mode,
            mask=mask
        )


    def _generate_sample(self, rng: Any) -> None:
        """
        Generate synthetic observed value by adding Gaussian noise to the latent sample.

        The result is stored in `self._sample`, which can be promoted to observation later.

        Raises:
            RuntimeError: If latent sample from input wave is missing.
        """

        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")

        noise = random_normal_array(x.shape, dtype=self.input_dtype, rng=rng)
        self._sample = x + np().sqrt(self._var) * noise

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the observation as an UncertainArray message.

        Depending on the precision mode, either scalar or array precision
        will be enforced before returning.

        Returns:
            UA: The observation in message form.
    
        Raises:
            RuntimeError: If no observation is set.
        """

        self._check_observed()
        if self.precision_mode_enum == PrecisionMode.SCALAR:
            return self.observed.as_scalar_precision()
        else:
            return self.observed

    def set_observed(self, data: np().ndarray, var: Optional[float] = None) -> None:
        """
        Manually assign the observation and precision (overrides current value).

        Args:
            data: Observed measurement array (same shape as input wave).
            var: Optional override for noise variance (defaults to internal value).
        """
        var = var if var is not None else self._var
        prec = 1.0 / var

        # --- dtype handling ---
        if self.expected_observed_dtype is None:
            self.expected_observed_dtype = data.dtype
        else:
            if data.dtype != self.expected_observed_dtype:
                data = data.astype(self.expected_observed_dtype)

        # --- consistency with input_dtype ---
        if self.input_dtype is not None:
            if np().issubdtype(self.input_dtype, np().floating):
                if not np().issubdtype(self.expected_observed_dtype, np().floating):
                    raise TypeError(
                        f"Observed dtype {self.expected_observed_dtype} must be real "
                        f"to match input dtype {self.input_dtype}"
                    )
            elif np().issubdtype(self.input_dtype, np().complexfloating):
                if not np().issubdtype(self.expected_observed_dtype, np().complexfloating):
                    raise TypeError(
                        f"Observed dtype {self.expected_observed_dtype} must be complex "
                        f"to match input dtype {self.input_dtype}"
                    )

        # --- precision handling ---
        if self._mask is not None:
            if data.shape != self._mask.shape:
                raise ValueError("Observed data and mask shape mismatch.")
            precision = np().where(self._mask, prec, 0.0)
        elif self.precision_mode_enum == PrecisionMode.SCALAR:
            precision = prec
        else:
            precision = np().full_like(data, fill_value=prec, dtype=get_real_dtype(self.expected_observed_dtype))

        self.observed = UA(data, dtype=self.expected_observed_dtype, precision=precision)


    def update_observed_from_sample(self) -> None:
        """
        Promote internal sample (from `generate_sample`) to actual observation.

        Constructs appropriate precision array using internal variance and mask.
    
        Raises:
            RuntimeError: If no internal sample is available.
        """

        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        prec = 1.0 / self._var

        if self._mask is not None:
            if self._sample.shape != self._mask.shape:
                raise ValueError("Sample and mask shape mismatch.")
            precision = np().where(self._mask, prec, 0.0)
        elif self.precision_mode_enum == PrecisionMode.SCALAR:
            precision = prec
        else:
            precision = np().full_like(self._sample, fill_value=prec, dtype=np().float64)

        self.observed = UA(self._sample, dtype=self.expected_observed_dtype, precision=precision)


    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen}, mode={self.precision_mode})"
