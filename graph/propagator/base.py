from abc import ABC, abstractmethod
from typing import Optional, Union

from ..factor import Factor
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import UnaryPropagatorPrecisionMode
from ...core.backend import np


class Propagator(Factor, ABC):
    """
    Abstract base class for deterministic mappings from input Waves to an output Wave.

    A `Propagator` represents a computational operator (e.g., addition, multiplication)
    that maps one or more input latent variables (`Wave` nodes) to a single output `Wave`.

    It serves as the core building block for composing computational factor graphs
    within the EP framework. Propagators manage both forward and backward message-passing,
    and coordinate precision mode inference across input/output nodes.

    Responsibilities:
        - Receive messages from inputs and compute output belief (forward)
        - Receive message from output and send approximate messages to inputs (backward)
        - Set precision mode for inputs/outputs based on known constraints

    Precision Mode Handling:
        - `UnaryPropagatorPrecisionMode` defines allowed configurations
          (e.g., scalar → array, array → scalar, etc.)
        - Precision propagation is handled via `set_precision_mode_forward()` and `backward()`
        - Conflicting precision assignments are explicitly rejected

    Subclass Requirements:
        - Must implement `_compute_forward()` and `_compute_backward()`
        - Must define `set_precision_mode_forward()` and `backward()`

    Args:
        input_names (tuple[str, ...]): Keys for input variables (e.g., ("a", "b")).
        dtype (np().dtype): Expected dtype for inputs/outputs (default: np().complex128).
        precision_mode (str | UnaryPropagatorPrecisionMode | None): Optional mode.

    Attributes:
        input_names (tuple[str, ...]): Keys identifying inputs.
        dtype (np().dtype): Common dtype used for messages.
        _precision_mode (UnaryPropagatorPrecisionMode | None): Precision configuration.
    """

    def __init__(
        self,
        input_names: tuple[str, ...] = ("input",),
        dtype: np().dtype = np().complex128,
        precision_mode: Optional[Union[str, UnaryPropagatorPrecisionMode]] = None,
    ):

        super().__init__()
        self.dtype = dtype
        self.input_names = input_names

        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

    def to_backend(self) -> None:
        """Synchronize dtype with current backend."""
        current_backend = np()
        if self.dtype is not None:
            self.dtype = current_backend.dtype(self.dtype)

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
