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
        var: float = 1.0,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        with_mask: bool = False
    ) -> None:
        self._var = var
        self._precision_value = 1.0 / var

        if isinstance(precision_mode, str):
            precision_mode = PrecisionMode(precision_mode)

        super().__init__(with_mask = with_mask)
        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

    def _infer_observed_dtype_from_input(self, input_dtype: np().dtype) -> np().dtype:
        # Use the same dtype for observed data as for latent variable
        return input_dtype

    def _compute_message(self, incoming: UA) -> UA:
        self._check_observed()
        if self.precision_mode_enum == PrecisionMode.SCALAR:
            return self.observed.as_scalar_precision()
        return self.observed

    def _generate_sample(self, rng: Any) -> None:
        """
        Draw synthetic sample: x + ε, where ε ~ Normal(0, var)
        """
        x = self.input.get_sample()
        if x is None:
            raise RuntimeError("Input sample not available.")
        noise = random_normal_array(x.shape, dtype=self.input_dtype, rng=rng)
        self._sample = (x + np().sqrt(self._var) * noise).astype(self.input_dtype)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        return f"GMeas(gen={gen}, mode={self.precision_mode})"
