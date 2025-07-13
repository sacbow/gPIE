from abc import ABC, abstractmethod
from typing import Optional
from graph.wave import Wave
from core.uncertain_array import UncertainArray

class Factor(ABC):
    """
    Abstract base class for factor nodes in the Computational Factor Graph (CFG).

    A Factor represents a transformation or probabilistic dependency between one or more 
    input Wave nodes and (optionally) an output Wave. Subclasses define the actual 
    message-passing logic for belief propagation.

    Typical subclasses include:
    - Prior: Defines a distribution over a latent variable (only output).
    - Propagator: Connects input(s) to output via forward model.
    - Measurement: Observes an input wave (no output).

    Attributes:
        inputs (dict): Mapping of input name → Wave instance.
        output (Wave or None): The output variable (if any).
        input_messages (dict): Messages received from input Waves.
        output_message (UncertainArray or None): Message from the output Wave.
        _generation (int or None): Scheduling index used during compilation.
        _precision_mode (str or None): "scalar" or "array", inferred during graph compilation.
    """

    def __init__(self):
        """
        Initialize an abstract Factor node in the factor graph.

        Each factor connects one or more input Waves to an optional output Wave,
        and passes messages between them during belief propagation.
        Subclasses (e.g., Prior, Propagator, Measurement) must implement forward/backward logic.
        """

        # Connected wave nodes
        self.inputs = dict()               # str -> Wave
        self.output = None                 # Single output wave or None

        # Messages from/to wave nodes
        self.input_messages = dict()       # Wave -> UncertainArray
        self.output_message = None         # UncertainArray or None

        # Optional scheduling index
        self._generation = None

        #precision mode
        self._precision_mode: Optional[str] = None  # 'scalar' or 'array'

    def _set_generation(self, gen: int):
        """Assign generation index for scheduling."""
        self._generation = gen
    
    @property
    def precision_mode(self) -> Optional[str]:
        """Public accessor for precision mode."""
        return self._precision_mode
    
    # Subclasses (e.g. Propagator) may override this method to allow additional modes.
    def _set_precision_mode(self, mode: str):
        """
        Set the precision mode for the factor.

        Args:
            mode (str): Either "scalar" or "array".

        Raises:
            ValueError: If conflicting mode is already set.
        """
        if mode not in ("scalar", "array"):
            raise ValueError(f"Invalid precision mode for Factor: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for {type(self).__name__}: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    
    def get_output_precision_mode(self) -> Optional[str]:
        """Return precision mode of the output Wave based on this factor's constraints."""
        return None  # override in subclass

    def get_input_precision_mode(self, wave: Wave) -> Optional[str]:
        """Return required mode for a specific input Wave."""
        return None  # override in subclass

    def set_precision_mode_forward(self):
        """Propagate from input to output (default: no-op)."""
        pass  # override in subclass

    def set_precision_mode_backward(self):
        """Propagate from output to input (default: no-op)."""
        pass  # override in subclass

    
    def add_input(self, name: str, wave: Wave):
        """
        Connect a wave as input to this factor.

        Args:
            name (str): Name/key for this input (e.g., "input", "a", "b", etc.)
            wave (Wave): The Wave instance to connect.
        """
        self.inputs[name] = wave
        self.input_messages[wave] = None
        wave.add_child(self)
    
    def connect_output(self, wave: Wave):
        """
        Connect a Wave as the output of this factor.
        Handles bidirectional linking and generation scheduling.

        Args:
            wave (Wave): The Wave object to connect as output.
        """
        # Set as this factor's output
        self.output = wave

        # Determine generation index: max of all input generations + 1
        max_gen = max((w._generation for w in self.inputs.values()
               if w is not None and w._generation is not None), default=0)

        self._set_generation(max_gen + 1)

        # Set wave's generation accordingly
        wave._set_generation(self._generation + 1)

        # Register this factor as the parent of the wave
        wave.set_parent(self)

    def receive_message(self, wave: Wave, message: UncertainArray):
        """
        Receive a message from a connected wave and store it.

        Args:
            wave (Wave): Sender of the message.
            message (UncertainArray): Incoming message.

        Raises:
            ValueError: If the wave is not connected to this factor.
        """
        if wave in self.inputs.values():
            self.input_messages[wave] = message
        elif wave == self.output:
            self.output_message = message
        else:
            raise ValueError("Received message from unconnected Wave.")
    
    def generate_sample(self):
        """
        Generate and set sample on the output Wave.
        Should be overridden by Prior and Propagator subclasses.
        """
        raise NotImplementedError("generate_sample not implemented for this Factor")
    
    @property
    def generation(self):
        return self._generation

    @abstractmethod
    def forward(self):
        """
        Send message(s) from this factor to its output wave.
        Should call output.receive_message(self, message).
        """
        pass

    @abstractmethod
    def backward(self):
        """
        Send message(s) from this factor to its input wave(s).
        Should call input.receive_message(self, message).
        """
        pass

    def __repr__(self):
        """
        Return a concise string representation for visualization and debugging.
        Example: SparsePrior(gen=1)
        """
        cls = type(self).__name__
        gen = self._generation if self._generation is not None else "-"
        return f"{cls}(gen={gen})"

