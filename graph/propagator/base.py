from abc import ABC, abstractmethod
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray as UA

class Propagator(Factor, ABC):
    def __init__(self, input_names=("input",), dtype=np.complex128):
        """
        Base class for propagators with one or more inputs and a single output.
        input_names: tuple of names for input Wave nodes (e.g., ("a", "b")).
        """
        super().__init__()
        self.dtype = dtype

        # Reserve input keys only; actual waves must be added via add_input()
        for name in input_names:
            self.inputs[name] = None

    def forward(self):
        """
        Compute and send message to output wave.
        Requires all input messages to be present.
        """
        if not all(self.inputs[name] for name in self.inputs):
            raise RuntimeError("Inputs not fully connected.")

        messages = {
            name: self.input_messages[self.inputs[name]]
            for name in self.inputs
        }

        if any(msg is None for msg in messages.values()):
            raise RuntimeError("Missing input message(s) for forward.")

        msg_out = self._compute_forward(messages)
        self.output_message = msg_out
        self.output.receive_message(self, msg_out)

    def backward(self):
        """
        Send messages to input waves based on output message.
        """
        if self.output_message is None:
            raise RuntimeError("Missing output message for backward.")

        for name, wave in self.inputs.items():
            msg = self._compute_backward(self.output_message, exclude=name)
            self.input_messages[wave] = msg
            wave.receive_message(self, msg)

    @abstractmethod
    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        pass

    @abstractmethod
    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        pass
