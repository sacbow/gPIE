from abc import ABC, abstractmethod
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray as UA

class Propagator(Factor, ABC):
    def __init__(self, input_names=("input",), dtype=np.complex128):
        """
        Base class for propagators with one or more inputs and a single output.
        
        Args:
            input_names (tuple of str): Names of input Wave nodes (e.g., ("a", "b")).
            dtype (np.dtype): Data type of the wave signals (typically complex128).
        """
        super().__init__()
        self.dtype = dtype
        self.input_names = input_names  # 接続は後で add_input で行う

    def forward(self):
        """
        Compute and send a message to the output wave.
        Requires all input messages to be present.
        """
        if not all(self.inputs.get(name) for name in self.input_names):
            raise RuntimeError("Inputs not fully connected.")

        messages = {
            name: self.input_messages[self.inputs[name]]
            for name in self.input_names
        }

        if any(msg is None for msg in messages.values()):
            raise RuntimeError("Missing input message(s) for forward.")

        msg_out = self._compute_forward(messages)
        self.output.receive_message(self, msg_out)


    def backward(self):
        """
        Send messages to input waves based on output message.
        """
        if self.output_message is None:
            raise RuntimeError("Missing output message for backward.")

        for name, wave in self.inputs.items():
            if wave is None:
                raise RuntimeError(f"Input wave '{name}' not connected.")
            msg = self._compute_backward(self.output_message, exclude=name)
            self.input_messages[wave] = msg
            wave.receive_message(self, msg)

    @abstractmethod
    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        pass

    @abstractmethod
    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        pass
