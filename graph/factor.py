from abc import ABC, abstractmethod
from typing import Optional, Union
from graph.wave import Wave
from core.uncertain_array import UncertainArray
from core.types import PrecisionMode


class Factor(ABC):
    """
    Abstract base class for factor nodes in the Computational Factor Graph (CFG).

    A `Factor` represents a probabilistic dependency or transformation between one or more
    latent variables (`Wave` nodes). It handles message passing for approximate inference
    using Expectation Propagation (EP).

    There are three typical subclasses:
        - Prior: Defines a prior distribution over a single variable (output only).
        - Propagator: Maps one or more input variables to an output via a forward model.
        - Measurement: Applies likelihood constraint to an observed variable (input only).

    Attributes:
        inputs (dict[str, Wave]):
            Mapping from input names (e.g., "a", "x") to connected Wave nodes.
        output (Optional[Wave]):
            Output wave connected to this factor. May be None.
        input_messages (dict[Wave, Optional[UncertainArray]]):
            Received messages from each input wave.
        output_message (Optional[UncertainArray]):
            Message received from the output wave.
        _generation (Optional[int]):
            Scheduling index (topological depth in the graph).
        _precision_mode (Optional[PrecisionMode]):
            Precision mode required by this factor (e.g., scalar or array).
    """

    def __init__(self):
        # Connected wave nodes
        self.inputs: dict[str, Wave] = {}
        self.output: Optional[Wave] = None

        # Messages
        self.input_messages: dict[Wave, Optional[UncertainArray]] = {}
        self.output_message: Optional[UncertainArray] = None

        # Scheduling & precision
        self._generation: Optional[int] = None
        self._precision_mode: Optional[PrecisionMode] = None

    def _set_generation(self, gen: int):
        """Set scheduling index (used during graph compilation)."""
        self._generation = gen

    @property
    def generation(self) -> Optional[int]:
        """Return topological scheduling index."""
        return self._generation

    @property
    def precision_mode(self) -> Optional[PrecisionMode]:
        """Return the current precision mode of the factor (scalar or array)."""
        return self._precision_mode

    def _set_precision_mode(self, mode: Union[str, PrecisionMode]):
        """
        Set the precision mode for the factor, with consistency checks.

        Args:
            mode (str | PrecisionMode): Either a string ("scalar", "array") or Enum value.

        Raises:
            ValueError: If the string is invalid or mode conflicts with existing value.
            TypeError: If input is not a valid type.
        """
        if isinstance(mode, str):
            try:
                mode = PrecisionMode(mode)
            except ValueError:
                raise ValueError(f"Invalid precision mode string: {mode}")

        if not isinstance(mode, PrecisionMode):
            raise TypeError(f"Invalid precision mode type: {type(mode)}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for {type(self).__name__}: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        """
        Override in subclasses to specify required precision mode for the output Wave.

        Returns:
            PrecisionMode or None
        """
        return None

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        """
        Override in subclasses to specify required precision mode for a given input Wave.

        Args:
            wave (Wave): Input wave to query.

        Returns:
            PrecisionMode or None
        """
        return None

    def set_precision_mode_forward(self):
        """
        Optionally propagate precision mode forward: from inputs to output.

        Used during graph compilation to coordinate Wave-Factor consistency.
        Override in subclasses.
        """
        pass

    def set_precision_mode_backward(self):
        """
        Optionally propagate precision mode backward: from output to inputs.

        Used during graph compilation to coordinate Wave-Factor consistency.
        Override in subclasses.
        """
        pass

    def add_input(self, name: str, wave: Wave):
        """
        Connect an input Wave to this factor under a given name.

        Args:
            name (str): Name/key to refer to this input (e.g., "x", "lhs").
            wave (Wave): Wave instance to connect.
        """
        self.inputs[name] = wave
        self.input_messages[wave] = None
        wave.add_child(self)

    def connect_output(self, wave: Wave):
        """
        Connect a Wave as the output of this factor.

        This sets the parent/child links and also updates generation indices.

        Args:
            wave (Wave): Output Wave node to connect.
        """
        self.output = wave
        max_gen = max(
            (w._generation for w in self.inputs.values() if w._generation is not None),
            default=0
        )
        self._set_generation(max_gen + 1)
        wave._set_generation(self._generation + 1)
        wave.set_parent(self)

    def receive_message(self, wave: Wave, message: UncertainArray):
        """
        Receive a message from a connected wave.

        Args:
            wave (Wave): The sender Wave.
            message (UncertainArray): The message to store.

        Raises:
            ValueError: If wave is not connected to this factor.
        """
        if wave in self.inputs.values():
            self.input_messages[wave] = message
        elif wave == self.output:
            self.output_message = message
        else:
            raise ValueError("Received message from unconnected Wave.")

    def generate_sample(self):
        """
        Generate a sample on the output Wave (if applicable).

        Subclasses like `Prior` or `Propagator`
        """