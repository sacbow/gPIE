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
    Abstract base class for measurement factors in a Computational Factor Graph.

    A Measurement connects a latent variable (Wave) to observed data.
    Unlike Prior, it has inputs but no output.

    Attributes:
        input_dtype (np.dtype): Required dtype for input wave.
        expected_observed_dtype (np.dtype | None): Optional constraint on observed data dtype.
        observed (UncertainArray | None): Noisy observed measurement.
        _sample (np.ndarray | None): Cached sample generated from the input wave.
        _mask (np.ndarray | None): Optional binary mask for valid observed regions.
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
        Connect this measurement to a wave via `@` operator.

        Args:
            wave: The wave to observe.

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

        Raises:
            RuntimeError: If no observation is available.
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
        Manually set the observed measurement and precision.

        Args:
            data: Observed data.
            precision: Inverse variance (float or array).
            dtype: Optional override of dtype.
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
        Use internal sample to generate observed data with internal variance and mask.
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
