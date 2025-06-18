from abc import ABC, abstractmethod
import numpy as np

from graph.factor import Factor
from graph.wave import Wave
from core.uncertain_array import UncertainArray


class Prior(Factor, ABC):
    def __new__(cls, shape, dtype=np.complex128):
        """
        Override instance creation to return the output Wave directly.
        This enables syntax like: X = GaussianPrior(...)  # X is a Wave
        """
        instance = super().__new__(cls)
        instance.__init__(shape, dtype)
        return instance.output

    def __init__(self, shape, dtype=np.complex128):
        """
        Initialize the prior factor and its associated output wave.
        """
        super().__init__()
        self.shape = shape
        self.dtype = dtype

        # Prior has no inputs
        self.set_generation(0)

        # Create output wave and assign generation
        self.output = Wave(shape, dtype)
        self.output.set_generation(self.generation + 1)

        # Register this factor as parent of the output wave (structure only)
        self.output.add_parent(self)

        # Store the most recent output message
        self.output_message = None

    def forward(self):
        """
        Send a message to the output wave.
        - On the first call, use a random initialization.
        - On subsequent calls, compute message from previous state.
        """
        if self.output_message is None:
            msg = UncertainArray.random(self.shape, dtype=self.dtype)
        else:
            msg = self._compute_message(self.output_message)

        self.output_message = msg
        self.output.receive_message(self, msg)

    def backward(self):
        """
        Prior does not process backward messages.
        """
        pass

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """
        Compute the outgoing message to the output wave.
        Must be implemented by subclass (e.g., GaussianPrior).
        """
        pass
