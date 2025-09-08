from __future__ import annotations
from numbers import Number
from typing import Union, Optional, Literal, overload

from .backend import np
from .types import ArrayLike, PrecisionMode, Precision, get_real_dtype
from .linalg_utils import reduce_precision_to_scalar, random_normal_array

from numpy.typing import NDArray
import warnings
from typing import TYPE_CHECKING
import numpy as _np

if TYPE_CHECKING:
    RealNDArray = NDArray[_np.float64]
    ComplexNDArray = NDArray[_np.complex128]


class UncertainArray:
    """
    UncertainArray

    Represents one or more independent Gaussian distributions used in Expectation Propagation
    on factor graphs. Each UncertainArray encodes:
    - Mean values: complex or real-valued data
    - Precision: scalar or array, representing inverse variance

    Core features:
    - Supports batched arrays: shape = (N, *event_shape)
    - Precision mode is automatically inferred during graph compilation
    - Integrates with Wave/Factor message passing in gPIE

    Typical operations:
    - Fusion (×): Combine two messages
    - Residual (÷): Subtract messages
    - Damping: Stabilize updates by interpolation
    - Conversion: `.as_scalar_precision()` / `.as_array_precision()`

    Precision model:

    In Expectation Propagation (EP), each UncertainArray represents a Gaussian distribution
    with mean and inverse variance ("precision"). gPIE supports two precision modes:

    - "scalar": Assumes that each entry in the array shares the same uncertainty (isotropic).
    - "array": Allows per-element uncertainty (anisotropic).

    The appropriate mode is automatically inferred at graph compile-time based on the factor graph
    structure and the requirements of individual Factor nodes. For example:

    - FFT propagators require one of the input/output Waves to use scalar precision
    to ensure tractable message passing in the Fourier domain.
    - Priors or measurements with spatially varying confidence naturally prefer array precision.

    Backend support:

    - Compatible with NumPy or CuPy
    - Convert with `.to_backend()`

    """


    def __init__(
        self,
        array: ArrayLike,
        dtype: np().dtype = None,
        precision: Precision = 1.0,
        *,
        batched: bool = True
    ) -> None:
        """
        Initialize an UncertainArray representing one or more Gaussian-distributed beliefs.

        This class supports both single and batched modes of operation:

        - If `batched=True` (default), the input array is interpreted as a batch of 
        independent Gaussian variables. The first axis is treated as batch dimension:
            - array.shape = (batch_size, *event_shape)
        - If `batched=False`, the input is interpreted as a single variable and is reshaped to:
            - array.shape → (1, *original_shape)
        allowing unified handling via batch-aware logic internally.

        Precision values (scalar or array) are automatically reshaped to be broadcast-compatible
        with the internal data representation.

        Args:
            array: Mean values of the belief. Can be 1D or N-D.
                - If `batched=True`: shape = (N, *D)
                - If `batched=False`: shape = (*D,), reshaped internally to (1, *D)
            dtype: Optional NumPy dtype. If None, inferred from array.
            precision: Inverse variance, either scalar or array. Must broadcast to `array`.
            batched: Whether to treat the first axis as batch. Default is True.

        Raises:
            ValueError: If precision is non-positive or incompatible in shape.
        """

        if dtype is None:
            arr = np().asarray(array)
        else:
            arr = np().asarray(array, dtype=dtype)

        if batched:
            self.data = arr
        else:
            self.data = arr.reshape((1,) + arr.shape)

        self.dtype = self.data.dtype
        self._set_precision_internal(precision, batched)
        self._scalar_precision = self.is_scalar(self._precision) or (
            batched and self._precision.shape == (self.batch_size,) + (1,) * len(self.event_shape)
        )


    
    def is_scalar(self, value):
        return isinstance(value, Number) or (
            hasattr(value, "shape") and value.shape == ()
        )

    def is_real(self) -> bool:
        return np().issubdtype(self.dtype, np().floating)

    def is_complex(self) -> bool:
        return np().issubdtype(self.dtype, np().complexfloating)

    def astype(self, dtype: np().dtype) -> UncertainArray:
        """
        Convert this UncertainArray to a different dtype, adjusting precision appropriately.

        - If dtype is unchanged, return self.
        - Complex → Real: extract real part, scale precision by 2, with warning.
        - Real → Complex: promote to complex type, scale precision by 0.5.
        - Complex → Complex (different precision): direct cast, precision unchanged.

        Args:
            dtype: Target NumPy dtype (e.g., np().float32, np().float64, np().complex64, np().complex128)

        Returns:
            New UncertainArray with converted data and adjusted precision.

        Raises:
            TypeError: If conversion between unsupported dtypes is attempted.
        """
        if dtype == self.dtype:
            return self

        # Complex → Real
        if np().issubdtype(dtype, np().floating) and self.is_complex():
            if not np().allclose(self.data.imag, 0):
                warnings.warn(
                    "Casting complex UncertainArray to real will discard imaginary part.",
                    category=UserWarning
                )
            real_data = self.data.real.astype(dtype)
            new_precision = 2.0 * self.precision(raw=True)
            return UncertainArray(real_data, dtype=dtype, precision=new_precision)

        # Real → Complex
        if np().issubdtype(dtype, np().complexfloating) and self.is_real():
            complex_data = self.data.astype(dtype)
            new_precision = 0.5 * self.precision(raw=True)
            return UncertainArray(complex_data, dtype=dtype, precision=new_precision)

        # Complex → Complex (precision change: e.g., complex64 ↔ complex128)
        if np().issubdtype(dtype, np().complexfloating) and self.is_complex():
            complex_data = self.data.astype(dtype)
            # Precision remains unchanged (variance unaffected by precision scaling)
            return UncertainArray(complex_data, dtype=dtype, precision=self.precision(raw=True))

        raise TypeError(f"Unsupported dtype conversion: {self.dtype} → {dtype}")

    @property
    def real(self) -> UncertainArray:
        """
        Return a real-valued UncertainArray corresponding to the real part of the data.

        - If already real, returns self.
        - If complex, returns a real UA with doubled precision.

        Raises:
            ValueError: If precision is invalid or data is malformed.
        """
        if self.is_real():
            return self
        real_data = self.data.real.astype(get_real_dtype(self.dtype))
        new_precision = 2.0 * self.precision(raw = True) # Complex → Real: Real part carries half the variance, so precision doubles
        return UncertainArray(real_data, dtype = get_real_dtype(self.dtype), precision=new_precision)


    def _set_precision_internal(self, value: Precision, batched: bool) -> None:
        """
        Internal setter for precision. Handles both scalar and array precision modes.

        This method sets `self._precision` to a broadcastable array representation,
        with the following conventions:

        - Scalar mode:
            If `value` is a float or scalar-like object, it is expanded to shape:
                (batch_size,) + (1,) * len(event_shape)
            This ensures it can be broadcast over `self.data` of shape:
                (batch_size, *event_shape)
            The flag `self._scalar_precision` is set to True.

        - Array mode:
            If `value` is an array, its shape must be broadcast-compatible with `self.data`.
            A special case is handled where a 1D array of shape (batch_size,) is reshaped to:
                (batch_size,) + (1,) * len(event_shape)
            The flag `self._scalar_precision` is set to False.

        Raises:
            ValueError: If any precision value is non-positive or the shape is not broadcastable.
        """

        real_dtype = get_real_dtype(self.dtype)

        if self.is_scalar(value):
            if value <= 0:
                raise ValueError("Precision must be positive.")
            
            # Compute minimal broadcastable shape for scalar precision
            if batched:
                # Shape: (batch_size,) + (1,)*len(event_shape)
                shape = (self.batch_size,) + (1,) * len(self.event_shape)
            else:
                # Shape: (1,)*ndim
                shape = (1,) * self.data.ndim
            
            self._precision = np().full(shape, float(value), dtype=real_dtype)
            self._scalar_precision = True

        else:
            # Convert to ndarray of proper dtype
            arr = (
                value if isinstance(value, np().ndarray) and value.dtype == real_dtype
                else np().asarray(value, dtype=real_dtype)
            )

            if batched and arr.ndim == 1 and arr.shape[0] == self.batch_size:
                arr = arr.reshape((self.batch_size,) + (1,) * len(self.event_shape))

            # Check broadcast compatibility
            try:
                np().broadcast_shapes(arr.shape, self.data.shape)
            except Exception:
                raise ValueError(
                    f"Precision shape {arr.shape} is not broadcastable to data shape {self.data.shape}."
                )
            
            self._precision = arr
            self._scalar_precision = False


    def precision(self, raw: bool = False) -> NDArray | float:
        """
        Return the precision (inverse variance) associated with this UncertainArray.

        Precision can be stored either in scalar mode (shared uncertainty across all entries)
        or array mode (per-element uncertainty). This method provides access to the underlying
        precision representation.

        Args:
            raw (bool): 
                - If False (default): Returns a broadcasted array of precision values 
                with shape equal to `self.data.shape`.
                - If True: Returns the internal representation of precision:
                    * Scalar mode: shape = (batch_size,) + (1, ...), i.e., minimal broadcastable shape
                    * Array mode: shape = self.data.shape

        Returns:
            np.ndarray or float: 
                - If `raw=False`: An array of shape `self.data.shape` for use in elementwise operations.
                - If `raw=True`: A compact representation suitable for broadcasting.
        """

        if raw:
            return self._precision
        else:
            return np().broadcast_to(self._precision, self.data.shape)

    
    def to_backend(self) -> None:
        """
        Convert `data` and `_precision` to current backend (NumPy/CuPy),
        handling explicit transfers between CPU (NumPy) and GPU (CuPy).
        """
        import cupy as cp
        current_backend = np()

        # --- data ---
        if isinstance(self.data, cp.ndarray) and current_backend.__name__ == "numpy":
            # CuPy → NumPy
            self.data = self.data.get()
        else:
            # NumPy → CuPy or same-backend
            self.data = current_backend.asarray(self.data)

        self.dtype = self.data.dtype

        # --- precision ---
        if isinstance(self._precision, cp.ndarray) and current_backend.__name__ == "numpy":
            # CuPy → NumPy
            self._precision = self._precision.get()
        else:
            # NumPy → CuPy or same-backend
            self._precision = current_backend.asarray(self._precision, dtype=get_real_dtype(self.dtype))


    @property
    def precision_mode(self) -> PrecisionMode:
        """
        Return the precision mode of this array.

        - 'scalar': single global precision used for all elements
        - 'array': elementwise precision used (per value)

        This property is determined automatically during construction.
        """
        return PrecisionMode.SCALAR if self._scalar_precision else PrecisionMode.ARRAY

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return (self.data.shape[0],) if self.batched else ()

    @property
    def batch_size(self) -> int:
        return self.data.shape[0] 

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.data.shape[1:]

    @property
    def shape(self) -> tuple[int, ...]:
        warnings.warn("UncertainArray.shape is deprecated. Use .event_shape instead.", DeprecationWarning, stacklevel=2)
        return self.data.shape

    @property
    def ndim(self) -> int:
        warnings.warn("UncertainArray.ndim is deprecated. Use len(.event_shape) instead.", DeprecationWarning, stacklevel=2)
        return self.data.ndim


    @classmethod
    def random(
        cls,
        event_shape: tuple[int, ...],
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision: float = 1.0,
        scalar_precision: bool = True,
        rng: Optional[Any] = None,
    ) -> "UncertainArray":
        """
        Create a random UncertainArray of shape (batch_size, *event_shape).

        Args:
            event_shape: Shape of each atomic variable (e.g., (64,), (32, 32)).
            batch_size: Number of instances (batchedd UA). Default: 1.
            dtype: Data type of the mean array. Default: complex64 (for GPU-friendliness).
            precision: Precision value (must be positive).
            scalar_precision: Whether to use scalar or elementwise precision.
            rng: Optional RNG. If None, fallback to default get_rng().

        Returns:
            UncertainArray: Randomly initialized UA with specified settings.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        shape = (batch_size,) + event_shape if batch_size > 1 else event_shape

        from .linalg_utils import random_normal_array
        data = random_normal_array(shape, dtype=dtype, rng=rng)

        if scalar_precision:
            return cls(data, dtype=dtype, precision=precision)
        else:
            arr_prec = np().full(shape, precision, dtype=get_real_dtype(dtype))
            return cls(data, dtype=dtype, precision=arr_prec)


    @classmethod
    def zeros(
        cls,
        event_shape: tuple[int, ...],
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision: float = 1.0,
        scalar_precision: bool = True,
    ) -> "UncertainArray":
        """
        Create a zero-filled UncertainArray of shape (batch_size, *event_shape).

        Args:
            event_shape: Shape of each atomic variable (e.g., (64,), (32, 32)).
            batch_size: Number of instances (batchedd UA). Default: 1.
            dtype: Data type of the mean array. Default: complex64.
            precision: Precision value (must be positive).
            scalar_precision: Whether to use scalar precision or elementwise.

        Returns:
            UncertainArray: Zero-filled data with specified precision mode.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        shape = (batch_size,) + event_shape if batch_size > 1 else event_shape

        data = np().zeros(shape, dtype=dtype)

        if scalar_precision:
            return cls(data, dtype=dtype, precision=precision)
        else:
            arr_prec = np().full(shape, precision, dtype=get_real_dtype(dtype))
            return cls(data, dtype=dtype, precision=arr_prec)

        
    def assert_compatible(self, other: "UncertainArray", context: str = "") -> None:
        """
        Ensure that another UncertainArray is compatible with self in structure:
        - batch_size
        - event_shape
        - dtype
        - precision mode

        Args:
            other: The UncertainArray to compare against.
            context: Optional context message (e.g., for debugging or error trace).

        Raises:
            ValueError: If shape or mode mismatch.
            TypeError: If dtype mismatch.
        """
        if self.batch_size != other.batch_size:
            raise ValueError(
                f"Batch size mismatch{f' in {context}' if context else ''}: "
                f"{self.batch_size} vs {other.batch_size}"
            )
        if self.event_shape != other.event_shape:
            raise ValueError(
                f"Event shape mismatch{f' in {context}' if context else ''}: "
                f"{self.event_shape} vs {other.event_shape}"
            )
        if self.dtype != other.dtype:
            raise TypeError(
                f"Dtype mismatch{f' in {context}' if context else ''}: {self.dtype} vs {other.dtype}"
            )
        if self.precision_mode != other.precision_mode:
            raise ValueError(
                f"Precision mode mismatch{f' in {context}' if context else ''}: "
                f"{self.precision_mode} vs {other.precision_mode}"
            )


    def __mul__(self, other: "UncertainArray") -> "UncertainArray":
        """
        Combine two UncertainArrays under the additive precision model.

        This corresponds to fusing two independent Gaussian beliefs:
            posterior_precision = p1 + p2
            posterior_mean = (p1 * m1 + p2 * m2) / (p1 + p2)
        """
        self.assert_compatible(other, context="__mul__")

        d1, d2 = self.data, other.data
        p1, p2 = self.precision(raw = True), other.precision(raw = True) 

        precision_sum = p1 + p2
        result_data = (p1 * d1 + p2 * d2) / precision_sum

        return UncertainArray(result_data, dtype=self.dtype, precision=precision_sum)

    
    def product_reduce_over_batch(self) -> "UncertainArray":
        """
        Reduce batchedd UA by fusing all atomic instances into one.

        This is equivalent to multiplying N Gaussians together:
            posterior_precision = sum_i p_i
            posterior_mean = sum_i (p_i * m_i) / sum_i p_i

        Returns:
            A new UncertainArray with batched=False.
        """

        # Get broadcasted precision of shape (N, *event_shape)
        precision = self.precision(raw = True)       # shape: (N, ...)
        weighted_data = precision * self.data  # shape: (N, ...)

        # Sum over batch axis (axis=0)
        precision_sum = np().sum(precision, axis=0)       # shape: event_shape
        weighted_data_sum = np().sum(weighted_data, axis=0)  # shape: event_shape
        reduced_data = np().divide(weighted_data_sum, precision_sum)

        return UncertainArray(reduced_data, dtype=self.dtype, precision=precision_sum, batched = False)


    def __truediv__(self, other: "UncertainArray") -> "UncertainArray":
        """
        Subtract a message from this UncertainArray under the additive precision model.

        This corresponds to computing a residual message in EP-style belief propagation:
            residual_precision = p1 - p2
            residual_mean = (p1 * m1 - p2 * m2) / max(p1 - p2, 1.0)
        """
        self.assert_compatible(other, context="__truediv__")

        d1, d2 = self.data, other.data
        p1, p2 = self.precision(raw = True), other.precision(raw = True)  # ← raw=False → shape == data.shape

        precision_diff = p1 - p2
        precision_safe = np().maximum(precision_diff, 1.0)  # ← element-wise safety

        result_data = (p1 * d1 - p2 * d2) / precision_safe

        return UncertainArray(result_data, dtype=self.dtype, precision=precision_safe)


    def damp_with(self, other: "UncertainArray", alpha: float) -> "UncertainArray":
        """
        Apply damping between this UncertainArray and another one.

        Performs convex interpolation of:
            - mean values (data)
            - standard deviation (not precision)

        This is used in EP/AMP-like updates where overcorrection is prevented.

        References:
            - Sarkar et al., "MRI Image Recovery using Damped Denoising Vector AMP", ICASSP 2021

        Args:
            other: Target UA to interpolate toward.
            alpha: Damping coefficient in [0, 1].

        Returns:
            New UA with damped mean and raw (unbroadcasted) precision.
        """
        self.assert_compatible(other, context="damp_with")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0, 1], but got {alpha}")

        # Interpolate means
        damped_data = (1 - alpha) * self.data + alpha * other.data

        # Interpolate standard deviations (use raw precision to preserve shape)
        std1 = np().sqrt(1.0 / self.precision(raw=True)).astype(get_real_dtype(self.dtype))
        std2 = np().sqrt(1.0 / other.precision(raw=True)).astype(get_real_dtype(self.dtype))
        damped_std = (1 - alpha) * std1 + alpha * std2
        damped_precision = 1.0 / (damped_std ** 2)

        return UncertainArray(damped_data, dtype=self.dtype, precision=damped_precision)


    def as_scalar_precision(self) -> "UncertainArray":
        """
        Convert to scalar precision mode (batch-wise harmonic reduction).
        """
        if self._scalar_precision:
            return self

        scalar_prec = reduce_precision_to_scalar(self.precision(raw=True))
        return UncertainArray(self.data, dtype=self.dtype, precision=scalar_prec)

    
    def as_array_precision(self) -> "UncertainArray":
        """
        Convert this UncertainArray to array precision mode.

        Promotes scalar precision to full per-element array.
        """
        if not self._scalar_precision:
            return self

        # Extract scalar value (guaranteed to be broadcastable constant)
        raw_prec = self.precision(raw=True)

        scalar_value = raw_prec.reshape((self.batch_size,) + (1,) * len(self.event_shape))

        array_precision = np().full(self.data.shape, scalar_value, dtype=get_real_dtype(self.dtype))

        return UncertainArray(self.data, dtype=self.dtype, precision=array_precision)


    def __repr__(self) -> str:
        """
        Return a string summary of the UncertainArray including event shape, batch size, and precision mode.
        Example:
            - UA(event_shape=(32, 32), precision=scalar)
            - UA(batch_size=10, event_shape=(64,), precision=array)
        """
        return (
                f"UA(batch_size={self.batch_size}, "
                f"event_shape={self.event_shape}, "
                f"precision={self.precision_mode}), "
                f"UA(..., dtype={self.dtype.name})"
            )