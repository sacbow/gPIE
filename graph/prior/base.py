from abc import ABC, abstractmethod
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray


class Prior(Factor, ABC):
    def __new__(cls, *args, **kwargs):
        """
        Override instance creation to return the output Wave directly.
        Accepts arbitrary arguments to pass to __init__.
        """
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance.output

    def __init__(self, shape, dtype=np.complex128, seed=None):
        """
        Initialize the prior factor and its associated output wave.
        """
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.seed = seed  # Optional deterministic random init

        # Create and connect output wave via unified method
        wave = Wave(shape, dtype)
        self.connect_output(wave)

    def forward(self):
        """
        Send a message to the output wave.
        - On the first call, use a random initialization (with optional seed).
        - On subsequent calls, compute message from previous state.
        """
        if self.output_message is None:
            msg = UncertainArray.random(self.shape, dtype=self.dtype, seed=self.seed)
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
