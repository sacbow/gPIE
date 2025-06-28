from abc import ABC, abstractmethod
from graph.wave import Wave
from core.uncertain_array import UncertainArray

class Factor(ABC):
    def __init__(self):
        # Connected wave nodes
        self.inputs = dict()               # str -> Wave
        self.output = None                 # Single output wave or None

        # Messages from/to wave nodes
        self.input_messages = dict()       # Wave -> UncertainArray
        self.output_message = None         # UncertainArray or None

        # Optional scheduling index
        self._generation = None

    def _set_generation(self, gen: int):
        """Assign generation index for scheduling."""
        self._generation = gen
    
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
        Receive a message from a connected Wave.
        Store it as input or output message depending on the source.
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

