from typing import Optional
import numpy as np
from core.linalg_utils import random_normal_array
from core.uncertain_array import UncertainArray
from core.uncertain_array_tensor import UncertainArrayTensor


class Wave:
    """
    A Wave represents a variable node in the Computational Factor Graph (CFG).

    Each Wave has one parent Factor (source of incoming messages) and any number 
    of child Factors (recipients of outgoing messages). Waves support forward 
    and backward message passing using Gaussian approximations (mean/precision).
    
    Attributes:
        shape (tuple): Shape of the underlying variable.
        dtype (np.dtype): Data type, typically np.complex128.
        label (str or None): Optional name used for identifying this node.
        parent (Factor or None): Upstream factor providing a message.
        children (list of Factor): Downstream factors receiving messages.
        _generation (int): Generation index for topological scheduling.
        _precision_mode (str or None): "scalar", "array", or None (auto-inferred).
        _sample (ndarray or None): Sampled value (optional).
    """

    def __init__(self, shape, dtype=np.complex128, precision_mode: Optional[str] = None, label = None):
        """
        Construct a new Wave node.

        Args:
            shape (tuple): Shape of the wave.
            dtype (np.dtype): Data type, typically complex-valued.
            precision_mode (str or None): Precision mode: "scalar" or "array".
            label (str or None): Optional label for graph-level reference.
        """

        self.shape = shape
        self.dtype = dtype
        self._precision_mode = precision_mode  # "scalar" or "array"
        self.label = label
        self._init_rng = None  # Will be set externally by Graph

        # Factor graph connections
        self.parent = None
        self.parent_message = None

        self.children = []

        # Belief is computed on demand
        self.belief = None

        # Generation index for scheduling
        self._generation = 0

        #random sample from probabilistic model
        self._sample = None
    
    def set_label(self, label: str):
        self.label = label

    def _set_generation(self, generation: int):
        """Internal method to assign generation index for inference scheduling."""
        self._generation = generation
    
    @property
    def precision_mode(self) -> Optional[str]:
        """Precision mode of the wave: 'scalar' or 'array'."""
        return self._precision_mode

    def _set_precision_mode(self, mode: str):
        """Set precision mode, and ensure consistency if already set."""
        if mode not in ("scalar", "array"):
            raise ValueError(f"Invalid precision mode: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for Wave: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    
    def set_precision_mode_forward(self):
        """Forward precision propagation based on parent factor."""
        if self.parent is not None:
            parent_mode = self.parent.get_output_precision_mode()
            if parent_mode is not None:
                self._set_precision_mode(parent_mode)

    def set_precision_mode_backward(self):
        """Backward precision propagation based on child factors."""
        for factor in self.children:
            child_mode = factor.get_input_precision_mode(self)
            if child_mode is not None:
                self._set_precision_mode(child_mode)


    def set_parent(self, factor):
        """Set the parent factor. Only one parent is allowed."""
        if self.parent is not None:
            raise RuntimeError("Parent factor is already set for this Wave.")
        self.parent = factor
        self.parent_message = None

    def add_child(self, factor):
        """Register a child factor and remember its index."""
        idx = len(self.children)
        self.children.append(factor)
        factor._child_index = idx  # optional: assign index to factor

    
    def finalize_structure(self):
        """
        After all children are added, initialize the child message tensor
        with random values and unit precision. The shape of precision is
        determined by this Wave's precision_mode.
        """
        n_child = len(self.children)
        shape = (n_child,) + self.shape
        data = random_normal_array(shape, dtype=self.dtype, rng=self._init_rng)

        if self._precision_mode == "scalar":
            # One scalar precision per message
            precision = np.ones(n_child, dtype=np.float64)  # shape (B,)
        else:
            # Elementwise precision per message
            precision = np.ones(shape, dtype=np.float64)    # shape (B, ...)

        self.child_messages_tensor = UncertainArrayTensor(data, precision, dtype=self.dtype)


    def receive_message(self, factor, message: UncertainArray):
        """
        Receive a message from a connected Factor.

        Depending on whether the factor is a parent or child, update the appropriate slot.

        Args:
            factor (Factor): The source of the message.
            message (UncertainArray): Incoming Gaussian approximation.

        Raises:
            TypeError: If dtype mismatch occurs.
            ValueError: If the factor is not recognized.
        """
        if message.dtype != self.dtype:
            raise TypeError(
                f"UncertainArray dtype {message.dtype} does not match Wave dtype {self.dtype}."
            )

        if factor == self.parent:
            self.parent_message = message
        elif factor in self.children:
            idx = self.children.index(factor)
            self.child_messages_tensor.data[idx] = message.data
            self.child_messages_tensor.precision[idx] = message._precision
        else:
            raise ValueError(
                f"Received message from unregistered factor: {factor}. "
                f"Expected parent: {self.parent}, or one of children: {list(self.child_messages)}"
            )

    def compute_belief(self):
        """
        Compute the current belief (posterior) at this Wave node.

        The belief is computed by fusing the incoming parent message
        (if present) with the combined message from all child factors.
        This uses additive precision combination (Gaussian fusion).

        Returns:
            UncertainArray: The updated belief of the node.
        """
        if self.parent_message is not None:
            combined = self.parent_message * self.child_messages_tensor.combine()
        else:
            combined = self.child_messages_tensor.combine()

        self.belief = combined
        return self.belief


    def forward(self):
        """
        Send forward messages to all child factors.

        This computes the current belief, removes the message from each child,
        and sends the resulting message to that child factor.
        """
        if self.parent_message is None:
            raise RuntimeError("Cannot forward without parent message.")
    
        if len(self.children) == 1:
            # Special case: single child — no need for belief computation or division
            factor = self.children[0]
            factor.receive_message(self, self.parent_message)
    
        else:
            belief = self.compute_belief()
            for i, factor in enumerate(self.children):
                msg = belief / self.child_messages_tensor[i]
                factor.receive_message(self, msg)


    def backward(self):
        """
        Send backward message to the parent factor.

        If there is only one child, its message is reused.
        Otherwise, all child messages are combined into one.
        """
        if self.parent is None:
            return

        if len(self.children) == 1:
            msg = self.child_messages_tensor[0]
        else:
            msg = self.child_messages_tensor.combine()

        self.parent.receive_message(self, msg)

    
    def set_init_rng(self, rng):
        """
        Set RNG for forward message initialization (random message at t=0).
        """
        self._init_rng = rng


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
    
    def __add__(self, other: "Wave") -> "Wave":
        """
        Define Z = X + Y as AddPropagator() @ (X, Y)
        """
        from graph.propagator.add_propagator import AddPropagator
        
        if not isinstance(other, Wave):
            raise TypeError("Can only add Wave to Wave.")
        return AddPropagator() @ (self, other)

    def __repr__(self):
        label_str = f", label='{self.label}'" if self.label else ""
        return f"Wave(shape={self.shape}, {label_str})"