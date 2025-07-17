from __future__ import annotations

from typing import Union, Optional, Literal, overload
from numpy.typing import NDArray
import numpy as np

from .linalg_utils import complex_normal_random_array, reduce_precision_to_scalar, random_normal_array

# Type aliases
ArrayLike = Union[NDArray[np.complex128], NDArray[np.float64]]
Precision = Union[float, NDArray[np.float64]]
PrecisionMode = Literal["scalar", "array"]


class UncertainArray:
    """
    A numerical array with associated scalar or elementwise precision,
    representing a belief/message in a probabilistic inference model.
    """

    def __init__(
        self,
        array: ArrayLike,
        dtype: np.dtype = np.complex128,
        precision: Precision = 1.0
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
        if np.issubdtype(dtype, np.floating) and self.is_complex():
            if not np.allclose(self.data.imag, 0):
                import warnings
                warnings.warn("Casting complex array to real will discard imaginary part.")
        return UncertainArray(
            array=self.data.astype(dtype),
            dtype=dtype,
            precision=self._precision
        )

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
        return "scalar" if self._scalar_precision else "array"

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

        Raises:
            TypeError: If dtype mismatch occurs.
        """

        if self.dtype != other.dtype:
            raise TypeError(f"Dtype mismatch: {self.dtype} vs {other.dtype}")

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

        For numerical stability:
            - Precision differences are clipped to be positive.
            - Scalar vs. array precision modes are handled accordingly.

        Args:
            other: The message to subtract (same dtype and shape).

        Returns:
            A new UncertainArray representing the residual message.

        Raises:
            TypeError: If dtype mismatch occurs.
        """

        if self.dtype != other.dtype:
            raise TypeError(f"Dtype mismatch: {self.dtype} vs {other.dtype}")

        d1, d2 = self.data, other.data
        p1, p2 = self.precision(raw=True), other.precision(raw=True)

        if self._scalar_precision and other._scalar_precision:
            precision_diff = p1 - p2
            precision_safe = max(precision_diff, 1e-12)
            result_data = (p1 * d1 - p2 * d2) / precision_safe
        else:
            p1_arr = self.precision()
            p2_arr = other.precision()
            precision_diff = p1_arr - p2_arr
            precision_safe = np.maximum(precision_diff, 1e-12)
            result_data = (p1_arr * d1 - p2_arr * d2) / precision_safe

        return UncertainArray(result_data, dtype=self.dtype, precision=precision_safe)

    def damp_with(self, other: UncertainArray, alpha: float) -> UncertainArray:
        """
        Apply damping between this UncertainArray and another one.

        This performs convex interpolation of both:
            - the mean values (data)
            - the **standard deviation** (not precision)

        The resulting UncertainArray reflects a softened update,
        useful for stabilizing message updates in iterative algorithms
        like EP or VAMP-style inference.

        Args:
            other: Target UncertainArray to interpolate toward.
            alpha: Damping coefficient in [0, 1]; 0 = keep self, 1 = replace with other.

        Returns:
            A new UncertainArray with damped mean and standard deviation.

        References:
            - Subrata Sarkar, Rizwan Ahmad, Philip Schniter,
            "MRI Image Recovery using Damped Denoising Vector AMP",
            ICASSP 2021. https://doi.org/10.1109/ICASSP39728.2021.9415050
        """

        if self.shape != other.shape:
            raise ValueError("Shape mismatch for damping.")
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
