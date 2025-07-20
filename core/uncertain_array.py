from __future__ import annotations

from typing import Union, Optional, Literal, overload
from core.types import PrecisionMode
from numpy.typing import NDArray
import numpy as np

from .linalg_utils import reduce_precision_to_scalar, random_normal_array

# Type aliases
ArrayLike = Union[NDArray[np.complex128], NDArray[np.float64]]
Precision = Union[float, NDArray[np.float64]]


class UncertainArray:
    """
    Represents a Gaussian-distributed random variable or message used in
    approximate inference algorithms (e.g., Expectation Propagation).

    Each UncertainArray stores:
        - Mean values (`data`) of the belief (real or complex array)
        - Precision (`precision`) representing inverse variance:
            * Scalar: global confidence level for all entries
            * Array: elementwise uncertainty per entry

    This abstraction enables flexible and numerically stable fusion of
    probabilistic messages in high-dimensional problems such as image
    reconstruction, inverse problems, or scientific inference.

    Typical use cases:
        - Representing priors, posteriors, or messages between nodes in a factor graph
        - Supporting scalar/array precision for uncertainty-aware computations
        - Enabling damping, fusion, and residual computations via ⨉ and ÷

    Examples:
        > ua1 = UncertainArray(np.zeros((32, 32)), precision=1.0)
        > ua2 = UncertainArray(np.ones((32, 32)), precision=np.ones((32, 32)))
        > ua_fused = ua1 * ua2
        > ua_residual = ua_fused / ua2
        > ua_damped = ua1.damp_with(ua2, alpha=0.3)

    Precision model:
        - `precision_mode` returns an Enum: PrecisionMode.SCALAR or ARRAY
        - Use `.precision(raw=True)` to get raw (float or ndarray) representation
        - Use `.precision()` to get broadcasted array of shape `data.shape`

    For scalar mode:
        precision ≈ float
        posterior_mean = weighted average
        posterior_precision = p1 + p2

    For array mode:
        precision ≈ NDArray[float64]
        computation done elementwise
    """


    def __init__(
        self,
        array: ArrayLike,
        dtype: np.dtype = np.complex128,
        precision: Precision = 1.0   # scalar precision implies self._scalar_precision = True.
    ) -> None:
        """
        Initialize an UncertainArray representing a belief/message.

        Args:
            array: Mean values of the belief (real or complex).
            dtype: Data type of the array (default: np.complex128).
            precision: Either a scalar precision (float) or elementwise precision (ndarray).
                    If scalar, the same precision is applied to all elements.

        Raises:
            ValueError: If precision is non-positive or shape mismatch occurs.
        """
        self.data: NDArray = np.asarray(array, dtype=dtype)
        self.dtype: np.dtype = self.data.dtype
        self._scalar_precision: bool = np.isscalar(precision) 
        self._set_precision_internal(precision)

    def is_real(self) -> bool:
        return np.issubdtype(self.dtype, np.floating)

    def is_complex(self) -> bool:
        return np.issubdtype(self.dtype, np.complexfloating)

    def astype(self, dtype: np.dtype) -> UncertainArray:
        """
        Convert this UncertainArray to a different dtype, adjusting precision appropriately.

        - If dtype is unchanged, return self.
        - Complex → Real: extract real part, scale precision by 2, with warning.
        - Real → Complex: promote to complex type, scale precision by 0.5.

        Args:
            dtype: Target NumPy dtype (np.float64 or np.complex128)

        Returns:
            New UncertainArray with converted data and adjusted precision.

        Raises:
            TypeError: If conversion between unsupported dtypes is attempted.
        """
        if dtype == self.dtype:
            return self

        if np.issubdtype(dtype, np.floating) and self.is_complex():
            if not np.allclose(self.data.imag, 0):
                import warnings
                warnings.warn(
                    "Casting complex UncertainArray to real will discard imaginary part.",
                    category=UserWarning
                )
            real_data = self.data.real.astype(dtype)
            new_precision = 2.0 * self.precision(raw = True)  # Complex → Real: Real part carries half the variance, so precision doubles
            return UncertainArray(real_data, dtype=dtype, precision=new_precision)

        if np.issubdtype(dtype, np.complexfloating) and self.is_real():
            complex_data = self.data.astype(dtype)
            new_precision = 0.5 * self.precision(raw = True)  # Real → Complex: Spreads variance over two dimensions, so precision halves
            return UncertainArray(complex_data, dtype=dtype, precision=new_precision)

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
        real_data = self.data.real.astype(np.float64)
        new_precision = 2.0 * self.precision(raw = True) # Complex → Real: Real part carries half the variance, so precision doubles
        return UncertainArray(real_data, dtype=np.float64, precision=new_precision)


    def _set_precision_internal(self, value: Precision) -> None:
        if np.isscalar(value):
            if value <= 0:
                raise ValueError("Precision must be positive.")
            self._precision: Precision = float(value)
        else:
            arr: NDArray[np.float64] = (
                value if isinstance(value, np.ndarray) and value.dtype == np.float64
                else np.asarray(value, dtype=np.float64)
            )
            if arr.shape != self.data.shape:
                raise ValueError("Precision array must match data shape.")
            self._precision = arr

    @overload
    def precision(self, raw: Literal[False] = False) -> NDArray[np.float64]: ...
    @overload
    def precision(self, raw: Literal[True]) -> Precision: ...

    def precision(self, raw: bool = False) -> Union[NDArray[np.float64], Precision]:
        """
        Access the precision (inverse variance) associated with this array.

        Args:
            raw: 
                - If False (default), returns a broadcasted ndarray matching `data.shape`.
                - If True, returns the raw internal value (`float` or `ndarray`) directly.

        Returns:
            Either a broadcasted precision array (when raw=False),
            or the internal precision representation (when raw=True).
        """
        if raw:
            return self._precision
        if self._scalar_precision:
            return np.full(self.shape, self._precision, dtype=np.float64)
        return self._precision

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
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @classmethod
    def random(
        cls,
        shape: tuple[int, ...],
        dtype: np.dtype = np.complex128,
        rng: Optional[np.random.Generator] = None,
        precision: float = 1.0,
        scalar_precision: bool = True
    ) -> UncertainArray:
        """
        Generate a random UncertainArray with normally distributed mean values
        and constant scalar or array precision.

        The mean (`data`) is sampled from:
            - N(0, 1) for real dtype
            - CN(0, 1) for complex dtype (i.e., standard complex normal)

        The precision is specified via:
            - `scalar_precision=True`: use a global scalar for all entries
            - `scalar_precision=False`: assign the same value to every element

        This is useful for:
            - initializing messages or beliefs before inference
            - generating synthetic uncertainty-aware inputs for testing

        Args:
            shape: Shape of the generated array (e.g., (64, 64)).
            dtype: Data type of the mean (np.complex128 or np.float64).
            rng: Optional NumPy random number generator.
            precision: Precision value to use (must be positive).
            scalar_precision: Whether to use scalar precision (default: True).

        Returns:
            UncertainArray: A random UA with given shape, dtype, and precision mode.

        Raises:
            ValueError: If precision is non-positive.

        Example:
            > ua = UncertainArray.random((128, 128), dtype=np.float64, precision=2.0)
            > ua.precision_mode
            <PrecisionMode.SCALAR: 'scalar'>
        """

        rng = np.random.default_rng() if rng is None else rng
        data = random_normal_array(shape, dtype=dtype, rng=rng)

        if scalar_precision:
            return cls(data, dtype=dtype, precision=precision)
        else:
            arr_precision = np.full(shape, precision, dtype=np.float64)
            return cls(data, dtype=dtype, precision=arr_precision)


    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        dtype: np.dtype = np.complex128,
        precision: float = 1.0,
        scalar_precision: bool = True
    ) -> UncertainArray:
        """
        Create an UncertainArray filled with zeros and specified precision.

        Args:
            shape: Shape of the array.
            dtype: Data type (default: np.complex128).
            precision: Precision value (float). Used for all entries.
            scalar_precision: If True, use scalar precision. If False, broadcast as array.

        Returns:
            UncertainArray with zero data and specified precision mode.
        """
        data = np.zeros(shape, dtype=dtype)

        if scalar_precision:
            return cls(data, dtype=dtype, precision=precision)
        else:
            arr_precision = np.full(shape, precision, dtype=np.float64)
            return cls(data, dtype=dtype, precision=arr_precision)
        
    def assert_compatible(self, other: UncertainArray, context: str = "") -> None:
        """
        Ensure that another UncertainArray is compatible with self in shape, dtype, and precision mode.

        Args:
            other: The UncertainArray to compare against.
            context: Optional context message (e.g., for debugging or error trace).

        Raises:
            ValueError: If shape mismatch.
            TypeError: If dtype mismatch.
            ValueError: If precision_mode mismatch.
        """
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch{f' in {context}' if context else ''}: {self.shape} vs {other.shape}")
        if self.dtype != other.dtype:
            raise TypeError(f"Dtype mismatch{f' in {context}' if context else ''}: {self.dtype} vs {other.dtype}")
        if self.precision_mode != other.precision_mode:
            raise ValueError(
                f"Precision mode mismatch{f' in {context}' if context else ''}: "
                f"{self.precision_mode} vs {other.precision_mode}"
            )


    def __mul__(self, other: UncertainArray) -> UncertainArray:
        """
        Combine two UncertainArrays under the additive precision model.

        This corresponds to fusing two independent Gaussian beliefs:
            posterior_precision = p1 + p2
            posterior_mean = (p1 * m1 + p2 * m2) / (p1 + p2)

        Args:
            other: Another UncertainArray with the same dtype and shape.

        Returns:
            A new UncertainArray representing the combined belief.
        """
        self.assert_compatible(other, context="__mul__")

        d1, d2 = self.data, other.data
        p1, p2 = self.precision(raw=True), other.precision(raw=True)
        precision_sum = p1 + p2
        result_data = (p1 * d1 + p2 * d2) / precision_sum

        return UncertainArray(result_data, dtype=self.dtype, precision=precision_sum)


    def __truediv__(self, other: UncertainArray) -> UncertainArray:
        """
        Subtract a message from this UncertainArray under the additive precision model.

        This corresponds to computing a residual message in EP-style belief propagation:
            residual_precision = p1 - p2
            residual_mean = (p1 * m1 - p2 * m2) / (p1 - p2)

        Args:
            other: The message to subtract (same dtype and shape).

        Returns:
            A new UncertainArray representing the residual message.
        """
        self.assert_compatible(other, context="__truediv__")

        d1, d2 = self.data, other.data
        p1, p2 = self.precision(raw=True), other.precision(raw=True)

        if self._scalar_precision and other._scalar_precision:
            precision_diff = p1 - p2
            precision_safe = max(precision_diff, 1.0)  # Avoid division by zero
            result_data = (p1 * d1 - p2 * d2) / precision_safe
        else:
            p1_arr = self.precision()
            p2_arr = other.precision()
            precision_diff = p1_arr - p2_arr
            precision_safe = np.maximum(precision_diff, 1.0)   # Avoid division by zero
            result_data = (p1_arr * d1 - p2_arr * d2) / precision_safe

        return UncertainArray(result_data, dtype=self.dtype, precision=precision_safe)


    def damp_with(self, other: UncertainArray, alpha: float) -> UncertainArray:
        """
        Apply damping between this UncertainArray and another one.

        This performs convex interpolation of both:
            - the mean values (data)
            - the **standard deviation** (not precision)

        References:
            - Subrata Sarkar, Rizwan Ahmad, Philip Schniter,
            "MRI Image Recovery using Damped Denoising Vector AMP", ICASSP 2021.

        Args:
            other: Target UncertainArray to interpolate toward.
            alpha: Damping coefficient in [0, 1].

        Returns:
            A new UncertainArray with damped mean and standard deviation.
        """
        self.assert_compatible(other, context="damp_with")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be in [0, 1].")

        damped_data = (1 - alpha) * self.data + alpha * other.data

        std1 = np.sqrt(1.0 / self.precision(raw=True))
        std2 = np.sqrt(1.0 / other.precision(raw=True))
        damped_std = (1 - alpha) * std1 + alpha * std2
        damped_precision = 1.0 / (damped_std ** 2)

        return UncertainArray(damped_data, dtype=self.dtype, precision=damped_precision)


    def as_scalar_precision(self) -> UncertainArray:
        """
        Convert this UncertainArray to scalar precision mode.

        This method reduces an elementwise precision array into a single scalar precision
        using the harmonic mean of variances:
            1 / mean(1 / p_i)

        This is useful for approximate inference algorithms that assume global precision,
        such as SparsePrior or when simplifying updates for efficiency.

        Returns:
            A new UncertainArray with the same mean values but scalar precision.

        Note:
            If the array is already in scalar mode, self is returned unchanged.
        """

        if self._scalar_precision:
            return self
        scalar_prec = reduce_precision_to_scalar(self._precision)
        return UncertainArray(self.data, dtype=self.dtype, precision=scalar_prec)

    def __repr__(self) -> str:
        """
        Return a string summary of the UncertainArray including shape and precision mode.
        Example: 'UA(shape=(32, 32), precision=array)'
        """
        return f"UA(shape={self.shape}, precision={self.precision_mode})"
