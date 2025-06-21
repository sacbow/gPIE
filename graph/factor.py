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
        self.generation = None

    def set_generation(self, gen: int):
        """Assign generation index for scheduling."""
        self.generation = gen
    
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
