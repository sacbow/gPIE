from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray


class Measurement(Factor, ABC):
    input_dtype: np.dtype  # Must be set in subclass
    expected_observed_dtype: Optional[np.dtype] = None  # Optionally set in subclass

    def __init__(self,
                 observed: Optional[UncertainArray] = None,
                 precision_mode: Optional[str] = None,
                 mask: Optional[np.ndarray] = None):
        """
        Initialize the Measurement base class.

        Args:
            input_wave (Wave): The Wave being measured.
            observed (UncertainArray or None): Optional observed data.
            precision_mode (str or None): "scalar", "array", or None.
            mask (ndarray of bool or None): Optional mask indicating valid observations.
        """
        if not hasattr(self, "input_dtype"):
            raise NotImplementedError("Subclasses must define input_dtype")

        if observed is not None and self.expected_observed_dtype is not None:
            if observed.dtype != self.expected_observed_dtype:
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype {self.expected_observed_dtype}, "
                    f"but received {observed.dtype}"
                )

        super().__init__()
        self._sample = None
        self.observed = observed
        self._mask = mask

        # Automatically infer precision_mode if mask is given
        if precision_mode is None and mask is not None:
            precision_mode = "array"

        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

        self.observed_dtype = (
            observed.dtype if observed is not None else
            self.expected_observed_dtype or self.input_dtype
        )
    
    def __matmul__(self, wave: Wave):
        if not hasattr(self, "input_dtype"):
            raise NotImplementedError("Subclasses must define input_dtype")

        if wave.dtype != self.input_dtype:
            raise TypeError(
                f"{type(self).__name__} expects input dtype {self.input_dtype}, "
                f"but received {wave.dtype}"
            )

        self.input = wave
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        if self.expected_observed_dtype is not None:
            self.observed_dtype = self.expected_observed_dtype
        else:
            self.observed_dtype = self.input_dtype

        return self


    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask
    
    def set_precision_mode_forward(self):
        """
        Forward precision propagation:
        Set this factor's precision_mode based on the connected input Wave.
        """
        if self.input.precision_mode is not None:
            self._set_precision_mode(self.input.precision_mode)

    def get_input_precision_mode(self, wave: Wave) -> Optional[str]:
        if wave != self.input:
            return None
        return self.precision_mode

    def forward(self):
        pass  # Measurement does not send forward messages

    def backward(self):
        self._check_observed()
        incoming = self.input_messages[self.input]
        msg = self._compute_message(incoming)
        self.input.receive_message(self, msg)

    def _check_observed(self):
        if self.observed is None:
            raise RuntimeError("Observed data is not set for this measurement.")

    def get_sample(self):
        return self._sample

    def set_sample(self, sample):
        if sample.shape != self.input.shape:
            raise ValueError(f"Sample shape mismatch: expected {self.input.shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self):
        self._sample = None

    def set_observed(self, data, precision, dtype=None):
        dtype = dtype or self.observed_dtype

        if self.expected_observed_dtype is not None and dtype != self.expected_observed_dtype:
            raise TypeError(
                f"{type(self).__name__} expects observed dtype {self.expected_observed_dtype}, "
                f"but got {dtype}"
            )

        if data.shape != self.input.shape or precision.shape != data.shape:
            raise ValueError("Observed data and precision must match input shape.")

        if self._mask is not None and self._mask.shape != data.shape:
            raise ValueError("Mask shape must match observed data shape.")

        self.observed = UncertainArray(data, dtype=dtype, precision=precision)
        self.observed_dtype = dtype

    def update_observed_from_sample(self):
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        var = getattr(self, "_var", 1.0)
        dtype = self.expected_observed_dtype or self.input_dtype

        if self._mask is not None:
            precision = np.where(self._mask, 1.0 / var, 0.0)
        else:
            precision = 1.0 / var

        self.observed = UncertainArray(self._sample, dtype=dtype, precision=precision)
        self.observed_dtype = dtype

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        pass
