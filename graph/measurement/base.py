from abc import ABC, abstractmethod
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray


class Measurement(Factor, ABC):
    def __init__(self, input_wave: Wave, observed: UncertainArray = None):
        super().__init__()
        self.observed = observed
        self._sample = None  # Store raw noisy sample

        # Connect input wave
        self.add_input("input", input_wave)
        self.input = input_wave  # Optional shortcut

        # Generation is one step after input's generation
        self._set_generation(input_wave._generation + 1)

    def forward(self):
        """
        Measurement does not propagate messages forward.
        """
        pass

    def backward(self):
        """
        Send a message to the input wave based on observation and previous message.
        """
        self._check_observed()
        incoming = self.input_messages[self.input]
        msg = self._compute_message(incoming)
        self.input.receive_message(self, msg)

    def _check_observed(self):
        """Ensure that observed is set before message passing."""
        if self.observed is None:
            raise RuntimeError("Observed data is not set for this measurement.")

    # === Sample-related methods ===
    def get_sample(self):
        """Return the generated noisy observation (if any)."""
        return self._sample

    def set_sample(self, sample):
        """Set the observed sample explicitly (for external control)."""
        if sample.shape != self.input.shape:
            raise ValueError(f"Sample shape mismatch: expected {self.input.shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self):
        """Clear the stored observation sample."""
        self._sample = None

    def set_observed(self, data, precision, dtype=np.complex128):
        """
        Set the observed UncertainArray directly from data and precision.
        """
        if data.shape != self.input.shape or precision.shape != data.shape:
            raise ValueError("Observed data and precision must match input shape.")
        self.observed = UncertainArray(data, dtype=dtype, precision=precision)

    def update_observed_from_sample(self):
        """
        Use the current sample (y = x + noise) to define the observed UncertainArray.
        Falls back on internal dtype and variance if observed is not already defined.
        """
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        if self.observed is not None:
            dtype = self.observed.dtype
            precision = self.observed._precision
        else:
            dtype = getattr(self, "_dtype", np.complex128)
            var = getattr(self, "_var", 1.0)
            precision = 1.0 / var

        self.observed = UncertainArray(self._sample, dtype=dtype, precision=precision)

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """
        Compute message from the measurement to the input wave.
        Must be implemented by subclass.
        """
        pass
