from abc import ABC, abstractmethod
from typing import Optional, Any
from ...core.backend import np
from ...core.rng_utils import get_rng

from ..factor import Factor
from ..wave import Wave
from ...core.uncertain_array import UncertainArray
from ...core.types import PrecisionMode

class Prior(Factor, ABC):
    """
    Abstract base class for prior factors in a Computational Factor Graph (CFG).

    A `Prior` defines the generative origin of a `Wave` node by specifying
    its initial distribution (e.g., standard Gaussian, sparse, structured).

    Responsibilities:
        - Owns and connects a single output Wave (no inputs)
        - Sends initial messages during the forward pass
        - Optionally refines messages if feedback is available (e.g., structured priors)
        - Manages precision mode (scalar/array) based on Wave or user input

    Message Passing Semantics:
        - forward(): Sends a randomly sampled or updated belief to the connected Wave
        - backward(): No operation (priors do not receive messages)

    Precision Handling:
        - If the precision_mode is not explicitly set, it is inferred from the connected Wave
        - Random samples are generated using `UncertainArray.random(...)`

    DSL Integration:
        Supports syntactic sugar:
            >> x = ~MyPrior(...)   # equivalent to: x = prior.output

    Attributes:
        shape (tuple[int, ...]): Shape of the variable to be generated.
        dtype (np().dtype): Data type (default: np().complex128).
        output (Wave): The Wave node this prior generates.
        _precision_mode (PrecisionMode | None): Precision mode, if set or inferred.
        _init_rng (np().random.Generator | None): RNG used for random message generation.
    """
    
    def __invert__(self) -> Wave:
        """
        Enable `x = ~MyPrior(...)` syntax for DSL-like expression.

        Returns:
            Wave: the output wave of this prior
        """
        return self.output

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np().dtype = np().complex128,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.shape: tuple[int, ...] = shape
        self.dtype: np().dtype = dtype
        self._init_rng: Optional[Any] = None

        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

        wave = Wave(shape, dtype=dtype, precision_mode=precision_mode, label=label)
        self.connect_output(wave)

    def set_precision_mode_backward(self) -> None:
        """
        If the output wave's mode is externally fixed, adopt it into this Prior.
        """
        if self.output.precision_mode_enum is not None:
            self._set_precision_mode(self.output.precision_mode_enum)

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        """
        Return this Prior's precision mode (as Enum).
        """
        return self._precision_mode

    def set_init_rng(self, rng: Optional[Any]) -> None:
        """
        Set RNG used for initial sampling of this prior.
        """
        self._init_rng = rng

    def forward(self) -> None:
        """
        Send the forward message to the output wave.

        If this is the first iteration, sample randomly;
        otherwise, update using _compute_message.
        """
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured for Prior.")

            if self._precision_mode is None:
                raise RuntimeError("Precision mode must be set before forward.")

            scalar_precision = self._precision_mode == PrecisionMode.SCALAR

            msg = UncertainArray.random(
                shape=self.shape,
                dtype=self.dtype,
                rng=self._init_rng,
                scalar_precision=scalar_precision,
            )
        else:
            msg = self._compute_message(self.output_message)

        self.output.receive_message(self, msg)

    def backward(self) -> None:
        """
        No backward message from Prior.
        """
        pass

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """
        Compute the message based on incoming observation (used in structured priors).
        """
        pass
