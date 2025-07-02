import numpy as np
from core.uncertain_array import UncertainArray


class Wave:
    def __init__(self, shape, dtype=np.complex128):
        """
        Initialize a Wave node in the factor graph.

        Args:
            shape (tuple): Shape of the associated array.
            dtype (np.dtype): Data type of the wave's values (default: np.complex128).
        """
        self.shape = shape
        self.dtype = dtype

        # Factor graph connections
        self.parent = None
        self.parent_message = None

        self.children = []
        self.child_messages = dict()

        # Belief is computed on demand
        self.belief = None

        # Generation index for scheduling
        self._generation = 0

        #random sample from probabilistic model
        self._sample = None

    def _set_generation(self, generation: int):
        """Internal method to assign generation index for inference scheduling."""
        self._generation = generation


    def set_parent(self, factor):
        """Set the parent factor. Only one parent is allowed."""
        if self.parent is not None:
            raise RuntimeError("Parent factor is already set for this Wave.")
        self.parent = factor
        self.parent_message = None

    def add_child(self, factor):
        """Register a child factor (without message)."""
        self.children.append(factor)
        self.child_messages[factor] = None

    def receive_message(self, factor, message: UncertainArray):
        """
        Receive a message from a connected Factor.
        This updates either parent_message or a child_message depending on source.

        Ensures that the dtype of the incoming message matches the Wave's dtype.
        """
        if message.dtype != self.dtype:
            raise TypeError(
                f"UncertainArray dtype {message.dtype} does not match Wave dtype {self.dtype}."
            )

        if factor == self.parent:
            self.parent_message = message
        elif factor in self.child_messages:
            self.child_messages[factor] = message
        else:
            raise ValueError(
                f"Received message from unregistered factor: {factor}. "
                f"Expected parent: {self.parent}, or one of children: {list(self.child_messages)}"
            )

    def compute_belief(self):
        """
        Explicitly compute and store the current belief
        by combining all incoming messages.
        """
        messages = []
        if self.parent_message is not None:
            messages.append(self.parent_message)
        messages.extend(
            msg for msg in self.child_messages.values() if msg is not None
        )

        if not messages:
            raise ValueError("No messages available to compute belief.")

        self.belief = UncertainArray.combine(messages)
        return self.belief

    def forward(self):
        """
        Propagate a message from this wave to each child factor.

        If only one child exists, the message from the parent is
        passed through directly for efficiency.
        """
        if self.parent_message is None:
            raise RuntimeError("Cannot forward without parent message.")

        if len(self.children) == 1:
            factor = self.children[0]
            factor.receive_message(self, self.parent_message)
            return

        messages = [
            self.parent_message
        ] + [msg for msg in self.child_messages.values() if msg is not None]
        belief = UncertainArray.combine(messages)

        for factor in self.children:
            if self.child_messages[factor] is not None:
                msg = belief / self.child_messages[factor]
                factor.receive_message(self, msg)

    def backward(self):
        """
        Propagate a message from this wave to the parent factor.
        If only one child exists and its message is available, it is passed directly.
        Otherwise, combine all valid messages.
        """
        if self.parent is None:
            return

        if len(self.children) == 1:
            factor = self.children[0]
            msg = self.child_messages.get(factor)
            if msg is not None:
                self.parent.receive_message(self, msg)
                return
            else:
                raise RuntimeError("Single child message is missing.")

        # For multiple children
        valid_msgs = [msg for msg in self.child_messages.values() if msg is not None]
        if not valid_msgs:
            raise RuntimeError("No child messages available for backward propagation.")

        msg = UncertainArray.combine(valid_msgs)
        self.parent.receive_message(self, msg)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the wave (i.e., shape length)."""
        return len(self.shape)
    
    @property
    def generation(self):
        return self._generation
    
    #sampling-related methods
    def get_sample(self):
        """Return the stored sample value (if any)."""
        return self._sample
    
    def set_sample(self, sample):
        """Set the sample value explicitly."""
        if sample.shape != self.shape:
            raise ValueError(f"Sample shape mismatch: expected {self.shape}, got {sample.shape}")
        self._sample = sample
    
    def clear_sample(self):
        """Clear the stored sample (set to None)."""
        self._sample = None

    def __repr__(self):
        return f"Wave(gen={self._generation})"

