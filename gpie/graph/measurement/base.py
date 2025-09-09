from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from ...core.backend import np, move_array_to_current_backend
from ...core.types import PrecisionMode, get_real_dtype, get_complex_dtype
from ...core.uncertain_array import UncertainArray
from ..factor import Factor
from ..wave import Wave
from ..structure.graph import Graph


class Measurement(Factor, ABC):
    """
    Abstract base class for measurement factors in a Computational Factor Graph (CFG).

    A Measurement represents the link between a latent variable (`Wave`) and
    an observed noisy measurement. It plays the role of a likelihood term
    in probabilistic modeling.
    """

    expected_input_dtype: Optional[Any] = None     # np.floating or np.complexfloating
    expected_observed_dtype: Optional[Any] = None  # np.floating or np.complexfloating

    def __init__(
        self,
        observed: Optional[UncertainArray] = None,
        precision_mode: Optional[Union[str, PrecisionMode]] = None,
        mask: Optional[np().ndarray] = None,
    ) -> None:
        super().__init__()

        self._sample: Optional[np().ndarray] = None
        self.observed: Optional[UncertainArray] = observed
        self._mask: Optional[np().ndarray] = mask
        self.label: Optional[str] = None

        self.input_dtype: Optional[np().dtype] = None
        self.observed_dtype: Optional[np().dtype] = None

        # Precision mode inference from mask (if not explicitly set)
        if precision_mode is None and mask is not None:
            precision_mode = PrecisionMode.ARRAY
        if precision_mode is not None:
            self._set_precision_mode(precision_mode)

        # Observed type is not finalized until __matmul__, but we can type-check now
        if observed is not None and self.expected_observed_dtype is not None:
            if not np().issubdtype(observed.dtype, self.expected_observed_dtype):
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype compatible with "
                    f"{self.expected_observed_dtype}, but received {observed.dtype}"
                )

    def __matmul__(self, wave: Wave) -> "Measurement":
        self.input_dtype = wave.dtype

        if self.expected_input_dtype is not None:
            if not np().issubdtype(self.input_dtype, self.expected_input_dtype):
                raise TypeError(
                    f"{type(self).__name__} expects input dtype compatible with "
                    f"{self.expected_input_dtype}, but got {self.input_dtype}"
                )

        # Determine observed_dtype based on subclass logic
        self.observed_dtype = self._infer_observed_dtype_from_input(self.input_dtype)

        if self.expected_observed_dtype is not None:
            if not np().issubdtype(self.observed_dtype, self.expected_observed_dtype):
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype compatible with "
                    f"{self.expected_observed_dtype}, but inferred {self.observed_dtype}"
                )

        # Convert observed to desired dtype if needed
        if self.observed is not None and self.observed.dtype != self.observed_dtype:
            self.observed = self.observed.astype(self.observed_dtype)

        # Connect wave
        self.input = wave
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        # Register to active graph
        graph = Graph.get_active_graph()
        if graph is not None:
            label = getattr(self, "label", None)
            if label is not None:
                setattr(graph, label, self)
            else:
                i = 0
                while True:
                    candidate = f"measurement_{i}"
                    if not hasattr(graph, candidate):
                        setattr(graph, candidate, self)
                        self.label = candidate
                        break
                    i += 1

        return self

    def to_backend(self) -> None:
        if self.observed is not None:
            self.observed.to_backend()

        if self._mask is not None:
            self._mask = move_array_to_current_backend(self._mask, dtype=bool)

        self.input_dtype = move_array_to_current_backend(np().array(0, dtype=self.input_dtype)).dtype
        if self.expected_observed_dtype is not None:
            self.expected_observed_dtype = move_array_to_current_backend(
                np().array(0, dtype=self.expected_observed_dtype)
            ).dtype

    @property
    def mask(self) -> Optional[np().ndarray]:
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

    def forward(self) -> None:
        pass  # Measurement has no forward pass

    def backward(self) -> None:
        self._check_observed()
        incoming = self.input_messages[self.input]
        msg = self._compute_message(incoming)
        self.input.receive_message(self, msg)

    def _check_observed(self) -> None:
        if self.observed is None:
            raise RuntimeError("Observed data is not set for this measurement.")

    def get_sample(self) -> Optional[np().ndarray]:
        return self._sample

    def set_sample(self, sample: np().ndarray) -> None:
        expected_shape = (self.input.batch_size,) + self.input.event_shape
        if sample.shape != expected_shape:
            raise ValueError(f"Sample shape mismatch: expected {expected_shape}, got {sample.shape}")
        self._sample = sample

    def clear_sample(self) -> None:
        self._sample = None

    def set_observed(
        self,
        data: np().ndarray,
        precision: Union[float, np().ndarray, None] = None,
        dtype: Optional[Any] = None,
        batched: bool = True,
    ) -> None:
        """
        Set observed data and its associated precision.

        Args:
            data (ndarray): Observed data array.
            precision (float | ndarray | None): Precision (inverse variance).
                If None, default precision is inferred from self._var.
            dtype (dtype, optional): Dtype to cast to. Defaults to self.observed_dtype.
            batched (bool): Whether input is batched. Defaults to True.
        """
        dtype = dtype or self.observed_dtype
        var = getattr(self, "_var", 1.0)
        prec = precision if precision is not None else 1.0 / var

        if not batched:
            data = data.reshape((1,) + data.shape)

        if self.expected_observed_dtype is not None:
            if not np().issubdtype(dtype, self.expected_observed_dtype):
                raise TypeError(
                    f"{type(self).__name__} expects observed dtype compatible with "
                    f"{self.expected_observed_dtype}, but got {dtype}"
                )

        expected_shape = (self.input.batch_size,) + self.input.event_shape
        if data.shape != expected_shape:
            raise ValueError(f"Observed data shape mismatch: expected {expected_shape}, got {data.shape}")
        if self._mask is not None and self._mask.shape != expected_shape:
            raise ValueError("Mask shape must match observed data shape.")

        # Auto-generate precision array if needed
        if isinstance(prec, float):
            if self._mask is not None:
                precision = np().where(self._mask, prec, 0.0)
            elif self.precision_mode_enum == PrecisionMode.SCALAR:
                precision = prec
            else:
                precision = np().full_like(data, fill_value=prec, dtype=get_real_dtype(dtype))
        else:
            precision = prec

        ua = UncertainArray(data, dtype=dtype, precision=precision)
        if ua.dtype != self.observed_dtype:
            ua = ua.astype(self.observed_dtype)

        self.observed = ua
        self.observed_dtype = ua.dtype


    def update_observed_from_sample(self) -> None:
        if self._sample is None:
            raise RuntimeError("No sample available to update observed.")

        var = getattr(self, "_var", 1.0)
        dtype = self.observed_dtype

        if self._mask is not None:
            precision = np().where(self._mask, 1.0 / var, 0.0)
        else:
            precision = 1.0 / var

        self.set_observed(self._sample, precision=precision, dtype=dtype)

    @abstractmethod
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        """
        Compute backward message based on current observation and incoming message.
        """
        pass

    @abstractmethod
    def _generate_sample(self, rng: Any) -> None:
        """
        Generate synthetic observed sample based on input latent variable.
        """
        pass

    @abstractmethod
    def _infer_observed_dtype_from_input(self, input_dtype: np().dtype) -> np().dtype:
        """
        Decide observed dtype based on input dtype. Must be implemented by subclasses.
        """
        pass
