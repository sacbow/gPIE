from typing import Optional, Union, Any

from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import random_normal_array
from ...core.types import PrecisionMode, get_real_dtype
from .base import Measurement


class GaussianMeasurement(Measurement):
    """
    Gaussian measurement model: y ~ N(x, var) or CN(x, var)

    This measurement assumes the observation y is the latent x plus Gaussian noise:
        y = x + ε,  ε ~ N(0, var) or CN(0, var)

    Supports real and complex types, scalar or array precision, and optional masking.
    """

    expected_input_dtype = None
    expected_observed_dtype = None

    def __init__(
        self,
        observed_array: Optional[np().ndarray] = None,
        var: float = 1.0,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        mask: Optional[np().ndarray] = None,
        dtype: Optional[np().dtype] = None,
        batched: bool = True,
    ) -> None:
        self._var = var
        self._precision_value = 1.0 / var

        # Dtype hinting: sets input_dtype and observed_dtype before __matmul__
        if dtype is not None:
            self.input_dtype = dtype  # used by __matmul__ if available

        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode)

        if mask is not None and precision_mode == PrecisionMode.SCALAR:
            raise ValueError("Masked observation requires array precision mode.")

        observed: Optional[UA] = None
        if observed_array is not None:
            # Precision setup
            if mask is not None:
                if observed_array.shape != mask.shape:
                    raise ValueError("Shape mismatch between observed_array and mask.")
                precision = np().where(mask, self._precision_value, 0.0)
            elif precision_mode == PrecisionMode.SCALAR or precision_mode is None:
                precision = self._precision_value
            else:
                precision = np().full_like(
                    observed_array,
                    fill_value=self._precision_value,
                    dtype=get_real_dtype(observed_array.dtype),
                )

            observed = UA(observed_array, dtype=observed_array.dtype, precision=precision, batched=batched)

        super().__init__(observed=observed, precision_mode=precision_mode, mask=mask)

    def _infer_observed_dtype_from_input(self, input_dtype: np().dtype) -> np().dtype:
        # Observed dtype should match latent dtype
        return input_dtype

    def _compute_message(self, incoming: UA) -> UA:
        self._check_observed()
        if self.precision_mode_enum == PrecisionMode.SCALAR:
            return self.observed.as_scalar_precision()
        return self.observed

    def _generate_sample(self, rng: Any) -> None:
        """
        Draw synthetic sample: x + ε
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")
        noise = random_normal_array(x.shape, dtype=self.input_dtype, rng=rng)
        self._sample = (x + np().sqrt(self._var) * noise).astype(self.input_dtype)

    def set_observed(self, data: np().ndarray, var: Optional[float] = None, batched: bool = True) -> None:
        """
        Manually provide observed data with optional noise variance.
        """
        var = var if var is not None else self._var
        prec = 1.0 / var

        # Ensure dtype compatibility
        if not np().issubdtype(data.dtype, self.observed_dtype):
            data = data.astype(self.observed_dtype)

        # Build precision array
        if self._mask is not None:
            if data.shape != self._mask.shape:
                raise ValueError("Observed data and mask shape mismatch.")
            precision = np().where(self._mask, prec, 0.0)
        elif self.precision_mode_enum == PrecisionMode.SCALAR:
            precision = prec
        else:
            precision = np().full_like(
                data, fill_value=prec, dtype=get_real_dtype(self.observed_dtype)
            )

        self.observed = UA(data, dtype=self.observed_dtype, precision=precision, batched=batched)

    def update_observed_from_sample(self) -> None:
        """
        Promote stored sample to observation, adding Gaussian noise.
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
            precision = np().full_like(
                self._sample, fill_value=prec, dtype=get_real_dtype(self.observed_dtype)
            )

        self.observed = UA(self._sample, dtype=self.observed_dtype, precision=precision, batched=True)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen}, mode={self.precision_mode})"
