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

    def __init__(self, shape, dtype=np.complex128):
        """
        Initialize the prior factor and its associated output wave.
        """
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self._init_rng = None  # will be set by Graph if needed

        wave = Wave(shape, dtype)
        self.connect_output(wave)

    def set_init_rng(self, rng):
        """
        Set the initial RNG used for message initialization.
        This method allows external control (e.g., from Graph).
        """
        self._init_rng = rng

    def forward(self):
        """
        Send a message to the output wave.
        """
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured for Prior.")
            msg = UncertainArray.random(self.shape, dtype=self.dtype, rng=self._init_rng)
        else:
            msg = self._compute_message(self.output_message)

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
