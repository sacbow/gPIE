from abc import ABC, abstractmethod
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray


class Measurement(Factor, ABC):
    def __init__(self, input_wave: Wave, observed: UncertainArray):
        """
        Initialize a measurement factor.

        Args:
            input_wave (Wave): The wave variable being observed.
            observed (UncertainArray): Observed data, including precision.
        """
        super().__init__()
        self.observed = observed

        # Register the input wave using the unified helper
        self.add_input("input", input_wave)
        self.input = input_wave  # for convenience access

        # Set generation after input's generation
        self.set_generation(self.input.generation + 1)

    def forward(self):
        """
        Measurement does not propagate messages forward.
        """
        pass

    def backward(self):
        """
        Send a message to the input wave based on observation and previous message.
        """
        incoming = self.input_messages[self.input]
        msg = self._compute_message(incoming)
        self.input.receive_message(self, msg)

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """
        Compute message from the measurement to the input wave.
        Must be implemented by subclass (e.g., GaussianMeasurement).
        """
        pass
