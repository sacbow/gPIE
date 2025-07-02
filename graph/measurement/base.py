from abc import ABC, abstractmethod
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray


class Measurement(Factor, ABC):
    input_dtype: np.dtype  # Must be set in subclass
    expected_observed_dtype: np.dtype = None  # Optionally set in subclass

    def __init__(self, input_wave: Wave, observed: UncertainArray = None):
        # Ensure input_dtype is defined in subclass
        if not hasattr(self, "input_dtype"):
            raise NotImplementedError("Subclasses must define input_dtype")

        # Validate input wave dtype
        if input_wave.dtype != self.input_dtype:
            raise TypeError(
                f"{type(self).__name__} expects input dtype {self.input_dtype}, "
                f"but received {input_wave.dtype}"
            )

        # Validate observed dtype if available
        if observed is not None and self.expected_observed_dtype is not None:
            if observed.dtype != self.expected_observed_dtype:
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype {self.expected_observed_dtype}, "
                    f"but received {observed.dtype}"
                )

        super().__init__()
        self.input = input_wave
        self.add_input("input", input_wave)
        self._sample = None
        self.observed = observed

        self.observed_dtype = (
            observed.dtype if observed is not None else
            self.expected_observed_dtype or self.input_dtype
        )

        self._set_generation(input_wave.generation + 1)

    def forward(self):
        """Measurement nodes do not send messages forward."""
        pass

    def backward(self):
        """Send message to input wave based on observed data and incoming message."""
        self._check_observed()
        incoming = self.input_messages[self.input]
        msg = self._compute_message(incoming)
        self.input.receive_message(self, msg)

    def _check_observed(self):
        """Raise error if observed data is not set."""
        if self.observed is None:
            raise RuntimeError("Observed data is not set for this measurement.")

    # === Sampling-related methods ===

    def get_sample(self):
        """Return stored raw sample (used to generate observation)."""
        return self._sample

    def set_sample(self, sample):
        """Manually set raw sample used to create observations."""
        if sample.shape != self.input.shape:
            raise ValueError(f"Sample shape mismatch: expected {self.input.shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self):
        """Clear stored sample."""
        self._sample = None

    def set_observed(self, data, precision, dtype=None):
        """
        Set the observed UncertainArray manually.
        dtype defaults to `self.observed_dtype` if not specified.
        """
        dtype = dtype or self.observed_dtype

        if self.expected_observed_dtype is not None and dtype != self.expected_observed_dtype:
            raise TypeError(
                f"{type(self).__name__} expects observed dtype {self.expected_observed_dtype}, "
                f"but got {dtype}"
            )

        if data.shape != self.input.shape or precision.shape != data.shape:
            raise ValueError("Observed data and precision must match input shape.")

        self.observed = UncertainArray(data, dtype=dtype, precision=precision)
        self.observed_dtype = dtype  # Update to ensure consistency

    def update_observed_from_sample(self):
        """
        Generate `self.observed` from stored sample and variance.
        Falls back to attributes `_var` and `expected_observed_dtype`.
        """
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        var = getattr(self, "_var", 1.0)
        precision = 1.0 / var
        dtype = self.expected_observed_dtype or self.input_dtype
        self.observed = UncertainArray(self._sample, dtype=dtype, precision=precision)
        self.observed_dtype = dtype

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """
        Compute the backward message from measurement to input wave.
        Must be implemented in subclasses.
        """
        pass