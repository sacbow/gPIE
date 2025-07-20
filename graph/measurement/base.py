from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np

from ..factor import Factor
from ..wave import Wave
from core.uncertain_array import UncertainArray
from core.types import PrecisionMode
from graph.structure.graph import Graph



class Measurement(Factor, ABC):
    """
    Abstract base class for measurement factors in a Computational Factor Graph (CFG).

    A Measurement represents the link between a latent variable (`Wave`) and
    an observed noisy measurement. It plays the role of a likelihood term
    in probabilistic modeling.

    Key characteristics:
        - One input wave (latent variable).
        - No output wave (terminal node).
        - Operates only in the backward direction (updates input from observations).
        - Supports optional masking (e.g., for missing/corrupted data).
        - Can handle scalar or array precision modes.

    Attributes:
        input_dtype (np.dtype):
            Required dtype for the input wave. Must be defined in subclass or __init__.
        expected_observed_dtype (np.dtype | None):
            Optional constraint on the dtype of observed data.
        observed (UncertainArray | None):
            The observed data as an UncertainArray (mean and precision).
        _sample (np.ndarray | None):
            Cached simulated sample from the latent variable, for use in forward modeling.
        _mask (np.ndarray | None):
            Optional boolean array masking valid observed entries (True = valid).
        label (str | None):
            Optional identifier registered to the graph when `@` operator is used.
    """


    input_dtype: np.dtype                          # must be defined in subclass or __init__
    expected_observed_dtype: Optional[np.dtype] = None  # can be defined in subclass

    def __init__(
        self,
        observed: Optional[UncertainArray] = None,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        if not hasattr(self, "input_dtype"):
            raise NotImplementedError("Subclasses must define `input_dtype`")

        if observed is not None and self.expected_observed_dtype is not None:
            if observed.dtype != self.expected_observed_dtype:
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype {self.expected_observed_dtype}, "
                    f"but received {observed.dtype}"
                )

        self._sample: Optional[np.ndarray] = None
        self.observed: Optional[UncertainArray] = observed
        self._mask: Optional[np.ndarray] = mask
        self.label: Optional[str] = None

        # infer precision mode from mask if needed
        if precision_mode is None and mask is not None:
            precision_mode = PrecisionMode.ARRAY
        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

        # inferred default observed dtype
        self.observed_dtype: np.dtype = (
            observed.dtype if observed is not None
            else self.expected_observed_dtype or self.input_dtype
        )

    def __matmul__(self, wave: Wave) -> "Measurement":
        """
        Connect this measurement node to the given wave.

        This is syntactic sugar for graph construction: `Y = MyMeasurement(...) @ X`.

        Also performs:
            - Input dtype check and assignment.
            - Generation assignment for scheduling.
            - Optional label registration to the active Graph (via context).

        Returns:
            self
        """

        # If input_dtype is not set (e.g., in GaussianMeasurement), infer from wave
        if getattr(self, "input_dtype", None) is None:
            self.input_dtype = wave.dtype

        if wave.dtype != self.input_dtype:
            raise TypeError(
                f"{type(self).__name__} expects input dtype {self.input_dtype}, "
                f"but received {wave.dtype}"
            )

        self.input = wave
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        # Set dtype for observations
        if self.expected_observed_dtype is not None:
            self.observed_dtype = self.expected_observed_dtype
        else:
            self.observed_dtype = self.input_dtype
        
        # Register to active graph if inside `with graph.observe()` block
        graph = Graph.get_active_graph()
        if graph is not None:
            label = getattr(self, "label", None)
            if label is not None:
                setattr(graph, label, self)
            else:
                # measurement_0, measurement_1, ...
                i = 0
                while True:
                    candidate = f"measurement_{i}"
                    if not hasattr(graph, candidate):
                        setattr(graph, candidate, self)
                        self.label = candidate 
                        break
                    i += 1

        return self


    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @property
    def precision_mode_enum(self) -> Optional[PrecisionMode]:
        return self._precision_mode

    @property
    def precision_mode(self) -> Optional[str]:
        return self._precision_mode.value if self._precision_mode else None

    def set_precision_mode_forward(self) -> None:
        if self.input.precision_mode_enum is not None:
            self._set_precision_mode(self.input.precision_mode_enum)

    def get_input_precision_mode(self, wave: Wave) -> Optional[str]:
        if wave != self.input:
            return None
        return self.precision_mode

    def forward(self) -> None:
        pass  # Measurement has no forward pass

    def backward(self) -> None:
        """
        Compute and send a backward message to the input wave.

        Requires that `self.observed` is set and valid. The message is computed
        using the subclass-defined `_compute_message()`.

        Raises:
            RuntimeError: If observed data is not yet set.
        """

        self._check_observed()
        incoming = self.input_messages[self.input]
        msg = self._compute_message(incoming)
        self.input.receive_message(self, msg)

    def _check_observed(self) -> None:
        if self.observed is None:
            raise RuntimeError("Observed data is not set for this measurement.")

    def get_sample(self) -> Optional[np.ndarray]:
        return self._sample

    def set_sample(self, sample: np.ndarray) -> None:
        if sample.shape != self.input.shape:
            raise ValueError(f"Sample shape mismatch: expected {self.input.shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self) -> None:
        self._sample = None

    def set_observed(
        self,
        data: np.ndarray,
        precision: Union[float, np.ndarray],
        dtype: Optional[np.dtype] = None
    ) -> None:
        """
        Manually provide the observed data and its precision.

        Useful for injecting experimental or synthetic measurements.

        Args:
            data: The observed values (same shape as input wave).
            precision: Inverse variance; scalar or per-element array.
            dtype: Optional override of dtype. Must match expected_observed_dtype.

        Raises:
            TypeError: If dtype does not match expected_observed_dtype.
            ValueError: If shape of data, precision, or mask are inconsistent.
        """

        dtype = dtype or self.observed_dtype

        if self.expected_observed_dtype is not None and dtype != self.expected_observed_dtype:
            raise TypeError(
                f"{type(self).__name__} expects observed dtype {self.expected_observed_dtype}, "
                f"but got {dtype}"
            )

        if data.shape != self.input.shape or (isinstance(precision, np.ndarray) and precision.shape != data.shape):
            raise ValueError("Observed data and precision must match input shape.")

        if self._mask is not None and self._mask.shape != data.shape:
            raise ValueError("Mask shape must match observed data shape.")

        self.observed = UncertainArray(data, dtype=dtype, precision=precision)
        self.observed_dtype = dtype

    def update_observed_from_sample(self) -> None:
        """
        Generate synthetic observations from the stored latent sample.

        Uses internal `_sample` (set externally) and synthetic noise based on `_var`.
        If a mask is defined, applies zero precision outside the mask.
        """

        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        var = getattr(self, "_var", 1.0)
        dtype = self.expected_observed_dtype or self.input_dtype

        if self._mask is not None:
            precision = np.where(self._mask, 1.0 / var, 0.0)
        else:
            precision = 1.0 / var

        self.observed = UncertainArray(self._sample, dtype=dtype, precision=precision)
        self.observed_dtype = dtype

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """
        Compute backward message based on current observation and incoming message.
        """
        pass
