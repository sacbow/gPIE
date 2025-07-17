from __future__ import annotations

from typing import Optional, Literal, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from core.linalg_utils import random_normal_array
from core.uncertain_array import UncertainArray
from core.uncertain_array_tensor import UncertainArrayTensor

if TYPE_CHECKING:
    from graph.propagator.add_propagator import AddPropagator
    from graph.propagator.multiply_propagator import MultiplyPropagator
    from graph.structure.graph import Factor  # Wave uses this in several places


class Wave:
    """
    Represents a latent variable node in a Computational Factor Graph (CFG),
    typically used for approximate inference using Expectation Propagation (EP).

    Each Wave node:
        - Receives a single message from a parent Factor (e.g., prior or propagator)
        - Sends messages to one or more child Factors (e.g., measurements or constraints)
        - Maintains belief as a fused Gaussian approximation (UncertainArray)

    Precision mode:
        - 'scalar': Assumes identical precision (inverse variance) for all entries
        - 'array': Allows elementwise precision per entry
        - If not specified during construction, mode will be auto-determined via `compile()`

    Attributes:
        shape (tuple[int, ...]): Shape of the underlying variable (e.g., image size).
        dtype (np.dtype): Data type (e.g., np.complex128 or np.float64).
        label (Optional[str]): Optional identifier for debugging or visualization.
        precision_mode (Optional[str]): 'scalar' or 'array'; can be set manually or inferred.

        parent (Optional[Factor]): Upstream factor sending a message.
        parent_message (Optional[UncertainArray]): Current message from parent.

        children (list[Factor]): Downstream factors receiving messages.
        child_messages_tensor (Optional[UncertainArrayTensor]): Batched messages from children.

        belief (Optional[UncertainArray]): Current fused belief.
        _sample (Optional[np.ndarray]): Optional sample drawn from the belief.

    Typical lifecycle:
        1. Create Wave with shape/dtype
        2. Attach factors via `set_parent` and `add_child`
        3. Auto-determine precision_mode via `set_precision_mode_forward/backward`
        4. Initialize messages via `finalize_structure`
        5. Participate in EP via `forward` and `backward`
    """


    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype = np.complex128,
        precision_mode: Optional[Literal["scalar", "array"]] = None,
        label: Optional[str] = None,
    ) -> None:
        self.shape: tuple[int, ...] = shape
        self.dtype: np.dtype = dtype
        self._precision_mode: Optional[Literal["scalar", "array"]] = precision_mode
        self.label: Optional[str] = label
        self._init_rng: Optional[np.random.Generator] = None

        self.parent: Optional["Factor"] = None
        self.parent_message: Optional[UncertainArray] = None
        self.children: list["Factor"] = []
        self.child_messages_tensor: Optional[UncertainArrayTensor] = None

        self.belief: Optional[UncertainArray] = None
        self._generation: int = 0
        self._sample: Optional[NDArray] = None

    def set_label(self, label: str) -> None:
        self.label = label

    def _set_generation(self, generation: int) -> None:
        self._generation = generation

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def precision_mode(self) -> Optional[Literal["scalar", "array"]]:
        return self._precision_mode

    def _set_precision_mode(self, mode: Literal["scalar", "array"]) -> None:
        if mode not in ("scalar", "array"):
            raise ValueError(f"Invalid precision mode: {mode}")
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for Wave: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        """
        Propagate precision mode from parent Factor to this Wave.

        This is used during graph compilation to ensure consistency
        between Wave and Factor precision types.
        """

        if self.parent is not None:
            parent_mode = self.parent.get_output_precision_mode()
            if parent_mode is not None:
                self._set_precision_mode(parent_mode)

    def set_precision_mode_backward(self) -> None:
        """
        Propagate precision mode constraints from child Factors.

        Each child factor may impose a requirement on the precision mode
        of its connected Wave inputs. This method collects and applies them.
        """

        for factor in self.children:
            child_mode = factor.get_input_precision_mode(self)
            if child_mode is not None:
                self._set_precision_mode(child_mode)

    def set_parent(self, factor: Factor) -> None:
        if self.parent is not None:
            raise RuntimeError("Parent factor is already set for this Wave.")
        self.parent = factor
        self.parent_message = None

    def add_child(self, factor: Factor) -> None:
        idx = len(self.children)
        self.children.append(factor)
        factor._child_index = idx  # optional

    def finalize_structure(self) -> None:
        """
        Allocate the child message tensor based on the number of children
        and the precision mode.

        This must be called after all child factors have been registered,
        typically as part of `Graph.compile()`.
        """

        n_child = len(self.children)
        shape = (n_child,) + self.shape
        data = random_normal_array(shape, dtype=self.dtype, rng=self._init_rng)

        if self._precision_mode == "scalar":
            precision = np.ones(n_child, dtype=np.float64)
        else:
            precision = np.ones(shape, dtype=np.float64)

        self.child_messages_tensor = UncertainArrayTensor(data, precision, dtype=self.dtype)

    def receive_message(self, factor: Factor, message: UncertainArray) -> None:
        """
        Receive a message (UncertainArray) from a connected Factor.

        This message is routed to either:
            - `parent_message`, if the source is the parent
            - `child_messages_tensor[i]`, if the source is a registered child

        Type/shape/mode consistency is enforced via UAT's setitem.

        Args:
            factor: Source Factor node.
            message: Incoming Gaussian message.

        Raises:
            TypeError: If dtype mismatch.
            ValueError: If the factor is not registered.
        """

        if message.dtype != self.dtype:
            raise TypeError(
                f"UncertainArray dtype {message.dtype} does not match Wave dtype {self.dtype}."
            )

        if factor == self.parent:
            self.parent_message = message
        elif factor in self.children:
            idx = self.children.index(factor)
            self.child_messages_tensor[idx] = message  # use __setitem__ with safety
        else:
            raise ValueError(
                f"Received message from unregistered factor: {factor}. "
                f"Expected parent: {self.parent}, or one of children: {list(self.children)}"
            )

    def compute_belief(self) -> UncertainArray:
        if self.parent_message is not None:
            combined = self.parent_message * self.child_messages_tensor.combine()
        else:
            combined = self.child_messages_tensor.combine()

        self.belief = combined
        return self.belief

    def set_belief(self, belief: UncertainArray) -> None:
        if belief.shape != self.shape:
            raise ValueError(f"Belief shape mismatch: expected {self.shape}, got {belief.shape}")
        if belief.dtype != self.dtype:
            raise ValueError(f"Belief dtype mismatch: expected {self.dtype}, got {belief.dtype}")
        self.belief = belief

    def forward(self) -> None:
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
        if self.parent is None:
            return

        if len(self.children) == 1:
            msg = self.child_messages_tensor[0]
        else:
            msg = self.child_messages_tensor.combine()

        self.parent.receive_message(self, msg)

    def set_init_rng(self, rng: np.random.Generator) -> None:
        self._init_rng = rng

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def get_sample(self) -> Optional[NDArray]:
        return self._sample

    def set_sample(self, sample: NDArray) -> None:
        if sample.shape != self.shape:
            raise ValueError(f"Sample shape mismatch: expected {self.shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self) -> None:
        self._sample = None

    def __add__(self, other: Wave) -> Wave:
        from graph.propagator.add_propagator import AddPropagator

        if not isinstance(other, Wave):
            raise TypeError("Can only add Wave to Wave.")
        return AddPropagator() @ (self, other)

    def __mul__(self, other: Wave) -> Wave:
        from graph.propagator.multiply_propagator import MultiplyPropagator

        if not isinstance(other, Wave):
            raise TypeError("Can only multiply Wave by Wave.")
        return MultiplyPropagator() @ (self, other)

    def __repr__(self) -> str:
        label_str = f", label='{self.label}'" if self.label else ""
        return f"Wave(shape={self.shape}{label_str})"
