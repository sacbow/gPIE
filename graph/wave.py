from __future__ import annotations

from typing import Optional, TYPE_CHECKING, Any
from ..core.rng_utils import get_rng
from ..core.backend import np
from ..core.types import ArrayLike, PrecisionMode, Precision
from ..core.linalg_utils import reduce_precision_to_scalar, random_normal_array
from numpy.typing import NDArray
from ..core.uncertain_array import UncertainArray
from ..core.uncertain_array_tensor import UncertainArrayTensor

if TYPE_CHECKING:
    from .propagator.add_propagator import AddPropagator
    from .propagator.multiply_propagator import MultiplyPropagator
    from .structure.graph import Factor


class Wave:
    """
    A latent variable node in a Computational Factor Graph (CFG),
    representing a Gaussian-distributed belief updated via message passing.

    Each Wave corresponds to a vector-shaped random variable in the model.
    It manages:
        - A belief (mean and precision)
        - Messages from a parent and multiple children
        - Precision mode (scalar or array)
        - Optional sample for generative use

    Message Passing Semantics:
        - forward(): sends belief / child_message to each child
        - backward(): sends combined child messages to the parent
        - Belief is updated as: belief = parent_message * combine(child_messages)

    Precision Modes:
        - 'scalar': Single scalar precision per UA
        - 'array': Elementwise precision (same shape as mean)
        Internally stored as `PrecisionMode` enum, externally exposed as str.

    Typical Usage:
        >> a = Wave((64, 64))
        >> b = Wave((64, 64))
        >> c = a + b  # Equivalent to AddPropagator() @ (a, b)

    Attributes:
        shape (tuple[int, ...]): Shape of the variable (excluding batch).
        dtype (np().dtype): Data type of values (e.g., np().complex128).
        label (str | None): Optional name identifier for graph visualization/debugging.
        parent (Factor | None): Connected parent factor node.
        children (list[Factor]): Connected child factors.
        belief (UncertainArray | None): Current belief estimate.
        parent_message (UncertainArray | None): Incoming message from parent.
        child_messages_tensor (UncertainArrayTensor | None): Incoming messages from children.
        _precision_mode (PrecisionMode | None): Internal precision mode.
    """
    __array_priority__ = 1000

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np().dtype = np().complex128,
        precision_mode: Optional[str | PrecisionMode] = None,
        label: Optional[str] = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self._precision_mode: Optional[PrecisionMode] = (
            PrecisionMode(precision_mode) if isinstance(precision_mode, str) else precision_mode
        )
        self.label = label
        self._init_rng: Optional[Any] = None

        self.parent: Optional["Factor"] = None
        self.parent_message: Optional[UncertainArray] = None
        self.children: list["Factor"] = []
        self.child_messages_tensor: Optional[UncertainArrayTensor] = None

        self.belief: Optional[UncertainArray] = None
        self._generation: int = 0
        self._sample: Optional[NDArray] = None
    
    def to_backend(self) -> None:
        """
        Convert internal arrays (e.g. child message tensor) to current backend.

        This should be called when switching from NumPy to CuPy or vice versa.
        """
        if self.child_messages_tensor is not None:
            self.child_messages_tensor.to_backend()
            self.dtype = self.child_messages_tensor.dtype  # Update dtype after backend cast

        if self.belief is not None:
            self.belief.to_backend()
            self.dtype = self.belief.dtype  # Update to match backend-updated belief

        if self.parent_message is not None:
            self.parent_message.to_backend()

        # Optional: sample を backend に移す（可能なら）
        if self._sample is not None:
            self._sample = np().asarray(self._sample)

    def set_label(self, label: str) -> None:
        """Assign label to this wave (for debugging or visualization)."""
        self.label = label

    def _set_generation(self, generation: int) -> None:
        """Internal: Assign scheduling generation index."""
        self._generation = generation

    @property
    def generation(self) -> int:
        """Topological generation index for inference scheduling."""
        return self._generation
    
    @property
    def precision_mode_enum(self) -> Optional[PrecisionMode]:
        """
        Return the internal precision mode as an Enum (recommended for new code).

        Returns:
            PrecisionMode or None
        """
        return self._precision_mode

    @property
    def precision_mode(self) -> Optional[str]:
        """
        Return the precision mode as a string ("scalar" or "array").

        This is kept for backward compatibility. Use `precision_mode_enum` for new code.

        Returns:
            "scalar", "array", or None
        """
        return self._precision_mode.value if self._precision_mode else None

    def _set_precision_mode(self, mode: str | PrecisionMode) -> None:
        """
        Set the precision mode for this wave, ensuring consistency if already set.

        Raises:
            ValueError: If conflicting precision mode already assigned.
        """
        if isinstance(mode, str):
            mode = PrecisionMode(mode)
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for Wave: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        """
        Infer precision mode from parent factor's output requirements.
        Called during graph compilation.
        """
        if self.parent is not None:
            parent_mode = self.parent.get_output_precision_mode()
            if parent_mode is not None:
                self._set_precision_mode(parent_mode)

    def set_precision_mode_backward(self) -> None:
        """
        Infer precision mode from child factors' input requirements.
        Called during graph compilation.
        """
        for factor in self.children:
            child_mode = factor.get_input_precision_mode(self)
            if child_mode is not None:
                self._set_precision_mode(child_mode)

    def set_parent(self, factor: Factor) -> None:
        """Assign a parent factor. Each wave can have only one parent."""
        if self.parent is not None:
            raise RuntimeError("Parent factor is already set for this Wave.")
        self.parent = factor
        self.parent_message = None

    def add_child(self, factor: Factor) -> None:
        """Register a child factor to this wave."""
        idx = len(self.children)
        self.children.append(factor)
        factor._child_index = idx  # optional usage

    def finalize_structure(self) -> None:
        """
        Allocate child message tensor using the wave's precision mode.

        This should be called once all children are connected (e.g. in Graph.compile()).
        """
        n_child = len(self.children)
        shape = (n_child,) + self.shape
        data = random_normal_array(shape, dtype=self.dtype, rng=self._init_rng)

        if self._precision_mode == PrecisionMode.SCALAR:
            precision = np().ones(n_child, dtype=np().float64)
        else:
            precision = np().ones(shape, dtype=np().float64)

        self.child_messages_tensor = UncertainArrayTensor(data, precision, dtype=self.dtype)

    def receive_message(self, factor: Factor, message: UncertainArray) -> None:
        """
        Receive a message from either the parent or a child.

        If the message's dtype does not match the Wave's dtype:
            - If Wave expects real and message is complex → apply UA.real
            - If Wave expects complex and message is real → apply UA.astype(complex)

        Raises:
            TypeError: If dtype mismatch cannot be reconciled.
            ValueError: If factor is not connected to this wave.
        """
        if message.dtype != self.dtype:
            if np().issubdtype(self.dtype, np().floating) and np().issubdtype(message.dtype, np().complexfloating):
                message = message.real  # Complex → Real
            elif np().issubdtype(self.dtype, np().complexfloating) and np().issubdtype(message.dtype, np().floating):
                message = message.astype(self.dtype)  # Real → Complex
            else:
                raise TypeError(
                    f"UncertainArray dtype {message.dtype} does not match Wave dtype {self.dtype}, "
                    f"and cannot be safely converted."
                )

        if factor == self.parent:
            self.parent_message = message
        elif factor in self.children:
            idx = self.children.index(factor)
            self.child_messages_tensor[idx] = message
        else:
            raise ValueError(
                f"Received message from unregistered factor: {factor}. "
                f"Expected parent: {self.parent}, or one of children: {list(self.children)}"
            )


    def compute_belief(self) -> UncertainArray:
        """
        Compute current belief by combining parent and child messages.

        Returns:
            Fused `UncertainArray` belief.
        """
        if self.parent_message is not None:
            combined = self.parent_message * self.child_messages_tensor.combine()
        else:
            combined = self.child_messages_tensor.combine()

        self.belief = combined
        return self.belief

    def set_belief(self, belief: UncertainArray) -> None:
        """Manually assign the belief (used in propagators with internal computation)."""
        if belief.shape != self.shape:
            raise ValueError(f"Belief shape mismatch: expected {self.shape}, got {belief.shape}")
        if belief.dtype != self.dtype:
            raise ValueError(f"Belief dtype mismatch: expected {self.dtype}, got {belief.dtype}")
        self.belief = belief

    def forward(self) -> None:
        """
        Send messages to all child factors using EP-style division.
        forward(): sends (belief / child_message) to each child
        Requires that parent message has already been received.
        """
        if self.parent_message is None:
            raise RuntimeError("Cannot forward without parent message.")

        if len(self.children) == 1:
            self.children[0].receive_message(self, self.parent_message)
        else:
            belief = self.compute_belief()
            for i, factor in enumerate(self.children):
                msg = belief / self.child_messages_tensor[i]
                factor.receive_message(self, msg)

    def backward(self) -> None:
        """
        Send message to parent by combining all child messages.
        backward(): sends combined(child_messages) to parent
        If there's only one child, reuse its message directly.
        """
        if self.parent is None:
            return

        if len(self.children) == 1:
            msg = self.child_messages_tensor[0]
        else:
            msg = self.child_messages_tensor.combine()

        self.parent.receive_message(self, msg)

    def set_init_rng(self, rng) -> None:
        """Set backend-agnostic random generator."""
        self._init_rng = rng


    @property
    def ndim(self) -> int:
        """Return number of dimensions of the wave variable."""
        return len(self.shape)
    
    
    def _generate_sample(self) -> None:
        """Pull sample from parent factor if not already set."""
        if self._sample is not None:
            return
        if self.parent and hasattr(self.parent, "get_sample_for_output"):
            sample = self.parent.get_sample_for_output()
            self.set_sample(sample)
    

    def get_sample(self) -> Optional[NDArray]:
        """Return the current sample (if set). To be deplicated."""
        return self._sample

    def set_sample(self, sample: NDArray) -> None:
        """Set sample value explicitly, with shape check."""
        if sample.shape != self.shape:
            raise ValueError(f"Sample shape mismatch: expected {self.shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self) -> None:
        """Clear the stored sample."""
        self._sample = None

    def __add__(self, other):
        """
        x + other → AddConstPropagator if other is scalar or ndarray,
                    otherwise AddPropagator
        """
        from .propagator.add_propagator import AddPropagator
        from .propagator.add_const_propagator import AddConstPropagator
        from numpy import ndarray

        if isinstance(other, Wave):
            return AddPropagator() @ (self, other)

        if np().isscalar(other) or isinstance(other, ndarray):
            return AddConstPropagator(const=other) @ self

        return NotImplemented

    def __radd__(self, other):
        """
        other + x → same as x + other
        """
        return self.__add__(other)


    def __mul__(self, other) -> Wave:
        """
        Overloaded elementwise multiplication.

        Supports:
            - Wave * Wave → MultiplyPropagator
            - Wave * ndarray/scalar → MultiplyConstPropagator

        Args:
            other (Wave | ndarray | scalar)

        Returns:
            Wave
        """
        from .propagator.multiply_const_propagator import MultiplyConstPropagator
        from .propagator.multiply_propagator import MultiplyPropagator

        if isinstance(other, Wave):
            return MultiplyPropagator() @ (self, other)
        elif isinstance(other, (int, float, complex, np().ndarray)):
            return MultiplyConstPropagator(other) @ self
        return NotImplemented

    def __rmul__(self, other) -> Wave:
        """
        Right-side multiplication.

        Supports:
            - scalar * Wave
            - ndarray * Wave

        Returns:
            Wave
        """
        return self.__mul__(other)

    def __repr__(self) -> str:
        label_str = f", label='{self.label}'" if self.label else ""
        dtype_str = f", dtype={np().dtype(self.dtype).name}" if self.dtype else ""
        precision_str = f", precision={self.precision_mode}" if self.precision_mode else ""
        return f"Wave(shape={self.shape}{label_str}{dtype_str}{precision_str})"



