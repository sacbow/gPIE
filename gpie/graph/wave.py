from __future__ import annotations
import warnings
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
        event_shape: tuple[int, ...],
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[str | PrecisionMode] = None,
        label: Optional[str] = None,
    ) -> None:
        """
        Initialize a Wave representing a vectorized latent variable node.

        Args:
            event_shape: Shape of each atomic variable (e.g. (64, 64)).
            batch_size: Number of vectorized instances. Default: 1 (non-vectorized).
            dtype: Data type of the variable. Default: complex64 (for GPU efficiency).
            precision_mode: 'scalar' or 'array', or corresponding PrecisionMode enum.
            label: Optional label for visualization or debugging.
        """
        self.event_shape = event_shape
        self.batch_size = batch_size
        self.dtype = dtype
        self._precision_mode: Optional[PrecisionMode] = (
            PrecisionMode(precision_mode) if isinstance(precision_mode, str) else precision_mode
        )
        self.label = label
        self._init_rng: Optional[Any] = None

        self.parent: Optional["Factor"] = None
        self.parent_message: Optional[UncertainArray] = None
        self.children: list["Factor"] = []

        self.child_messages: dict["Factor", UncertainArray] = {}

        self.belief: Optional[UncertainArray] = None
        self._generation: int = 0
        self._sample: Optional[NDArray] = None

    
    def to_backend(self) -> None:
        """
        Convert all internal UncertainArrays (parent, children, belief) to current backend.

        This should be called when switching between NumPy and CuPy backends.
        """
        # Convert child messages
        for msg in self.child_messages.values():
            msg.to_backend()

        # Convert belief if it exists
        if self.belief is not None:
            self.belief.to_backend()
            self.dtype = self.belief.dtype  # Ensure dtype consistency

        # Convert parent message if it exists
        if self.parent_message is not None:
            self.parent_message.to_backend()


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
            raise ValueError("Parent factor is already set for this Wave.")
        self.parent = factor
        self.parent_message = None

    def add_child(self, factor: Factor) -> None:
        """Register a child factor to this wave."""
        if factor in self.child_messages:
            raise ValueError(f"Factor {factor} already registered as child.")
        self.children.append(factor)
        self.child_messages[factor] = None

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
        # --- Dtype reconciliation ---
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

        # --- Assign message ---
        if factor == self.parent:
            self.parent_message = message
        elif factor in self.children:
            self.child_messages[factor] = message
        else:
            raise ValueError(
                f"Received message from unregistered factor: {factor}. "
                f"Expected parent: {self.parent}, or one of children: {list(self.children)}"
            )

    def combine_child_messages(self) -> UncertainArray:
        """
        Combine all messages from children into a single UncertainArray belief.

        This performs weighted averaging:
            - posterior_precision = sum_i p_i
            - posterior_mean = sum_i (p_i * m_i) / sum_i p_i

        Returns:
            UncertainArray: Combined belief from children.
        """
        if not self.child_messages:
            raise RuntimeError("No child messages to combine.")

        iterator = iter(self.child_messages.values())
        first = next(iterator)
        dtype = first.dtype
        vectorize = first.vectorize
        p = first.precision(raw=True)
        weighted = p * first.data

        for ua in iterator:
            if ua.vectorize != vectorize:
                raise ValueError("Mismatched vectorization across child messages.")
            p_i = ua.precision(raw=True)
            weighted += p_i * ua.data
            p += p_i

        mean = weighted / p
        return UncertainArray(mean, dtype=dtype, precision=p, vectorize=vectorize)
    

    def set_belief(self, belief: UncertainArray) -> None:
        """Manually assign the belief (used in propagators with internal computation)."""
        if belief.batch_size != self.batch_size:
            raise ValueError(f"Belief batch_size mismatch: expected {self.batch_size}, got {belief.batch_size}")
        if belief.event_shape != self.event_shape:
            raise ValueError(f"Belief shape mismatch: expected {self.event_shape}, got {belief.event_shape}")
        if belief.dtype != self.dtype:
            raise ValueError(f"Belief dtype mismatch: expected {self.dtype}, got {belief.dtype}")
        self.belief = belief


    def compute_belief(self) -> UncertainArray:
        """
        Compute current belief by combining parent and child messages.

        Returns:
            Fused `UncertainArray` belief.
        """
        child_belief = self.combine_child_messages()

        if self.parent_message is not None:
            combined = self.parent_message * child_belief
        else:
            combined = child_belief

        self.set_belief(combined)
        return combined


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
            msg = self.child_messages[self.children[0]]
        else:
            msg = self.combine_child_messages()

        self.parent.receive_message(self, msg)


    def set_init_rng(self, rng) -> None:
        """Set backend-agnostic random generator."""
        self._init_rng = rng

    @property
    def ndim(self) -> int:
        """
        Deprecated: Use `len(.event_shape)` instead.
        """
        warnings.warn("Wave.ndim is deprecated. Use len(.event_shape) instead.", DeprecationWarning, stacklevel=2)
        return len(self.event_shape)
    
    def _generate_sample(self, rng) -> None:
        """Pull sample from parent factor if not already set."""
        if self._sample is not None:
            return
        if self.parent and hasattr(self.parent, "get_sample_for_output"):
            sample = self.parent.get_sample_for_output(rng = rng)
            self.set_sample(sample)
    

    def get_sample(self) -> Optional[NDArray]:
        """Return the current sample (if set). To be deplicated."""
        return self._sample

    def set_sample(self, sample: NDArray) -> None:
        """Set sample value explicitly, with shape check."""
        expected_shape = (self.batch_size,) + self.event_shape
        if sample.shape != expected_shape:
            raise ValueError(f"Sample shape mismatch: expected {expected_shape}, got {sample.shape}")
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
        if isinstance(other, Wave):
            return AddPropagator() @ (self, other)

        if np().isscalar(other) or isinstance(other, np().ndarray):
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

        if self.batch_size == 1:
            return f"Wave(event_shape={self.event_shape}{precision_str}{label_str}{dtype_str})"
        else:
            return (
                f"Wave(batch_size={self.batch_size}, "
                f"event_shape={self.event_shape}"
                f"{precision_str}{label_str}{dtype_str})"
            )