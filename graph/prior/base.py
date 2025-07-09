from abc import ABC, abstractmethod
import numpy as np
from typing import Literal, Optional

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray


class Prior(Factor, ABC):
    def __new__(cls, *args, **kwargs):
        """
        When instantiated, immediately return the associated output Wave.
        """
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        return instance.output

    def __init__(
        self,
        shape,
        dtype=np.complex128,
        precision_mode: Optional[Literal["scalar", "array"]] = None,
    ):
        """
        Initialize the Prior and its output Wave.

        Args:
            shape (tuple): Shape of the latent variable.
            dtype (np.dtype): Data type of the variable.
            precision_mode (str or None): "scalar", "array", or None.
        """
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self._init_rng = None
        self.precision_mode = precision_mode  # This affects the wave

        # Create Wave (may propagate mode later)
        wave = Wave(shape, dtype=dtype, precision_mode=precision_mode)
        self.connect_output(wave)

    def set_precision_mode_forward(self):
        """
        Propagate precision mode forward from the prior to its output wave.
        """
        if self.precision_mode is not None:
            self.output._set_precision_mode(self.precision_mode)

    def get_output_precision_mode(self) -> Optional[str]:
        """
        Return the preferred mode for the output wave.
        """
        return self.precision_mode

    def set_init_rng(self, rng):
        self._init_rng = rng

    def forward(self):
        """
        Send message to the output wave (initial message or updated).
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
        No backward message from Prior.
        """
        pass

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        pass
