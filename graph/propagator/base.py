from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray as UA
from core.types import UnaryPropagatorPrecisionMode


class Propagator(Factor, ABC):
    def __init__(
        self,
        input_names: tuple[str, ...] = ("input",),
        dtype: np.dtype = np.complex128,
        precision_mode: Optional[Union[str, UnaryPropagatorPrecisionMode]] = None,
    ):
        """
        Base class for propagators with one or more inputs and a single output.

        Args:
            input_names (tuple of str): Names of input Wave nodes (e.g., ("a", "b")).
            dtype (np.dtype): Data type of the wave signals.
            precision_mode (str | UnaryPropagatorPrecisionMode | None): Optional mode.
        """
        super().__init__()
        self.dtype = dtype
        self.input_names = input_names

        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

    @property
    def precision_mode_enum(self) -> Optional[UnaryPropagatorPrecisionMode]:
        return self._precision_mode

    @property
    def precision_mode(self) -> Optional[str]:
        return str(self._precision_mode) if self._precision_mode else None

    def _set_precision_mode(self, mode: Union[str, UnaryPropagatorPrecisionMode]) -> None:
        """
        Internal setter for precision_mode.

        Args:
            mode (str | UnaryPropagatorPrecisionMode): Mode to assign.

        Raises:
            ValueError: If mode is invalid or conflicts with existing mode.
        """
        if isinstance(mode, str):
            try:
                mode = UnaryPropagatorPrecisionMode(mode)
            except ValueError:
                raise ValueError(f"Invalid precision mode for Propagator: {mode}")

        if not isinstance(mode, UnaryPropagatorPrecisionMode):
            raise TypeError("Precision mode must be a string or UnaryPropagatorPrecisionMode")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for Propagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    @abstractmethod
    def set_precision_mode_forward(self) -> None:
        """Determine output wave mode based on inputs and self.precision_mode."""
        raise NotImplementedError("Propagator must define forward precision propagation.")

    @abstractmethod
    def set_precision_mode_backward(self) -> None:
        """Determine input wave modes based on output and self.precision_mode."""
        raise NotImplementedError("Propagator must define forward precision propagation.")

    def forward(self) -> None:
        """Compute and send a message to the output wave."""
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

    def backward(self) -> None:
        """Send messages to input waves based on output message."""
        if self.output_message is None:
            raise RuntimeError("Missing output message for backward.")

        for name, wave in self.inputs.items():
            if wave is None:
                raise RuntimeError(f"Input wave '{name}' not connected.")
            msg = self._compute_backward(self.output_message, exclude=name)
            self.input_messages[wave] = msg
            wave.receive_message(self, msg)

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        raise NotImplementedError("This propagator does not implement _compute_forward.")

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        raise NotImplementedError("This propagator does not implement _compute_backward.")
