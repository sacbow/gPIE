import numpy as np
from .linalg_utils import complex_normal_random_array, reduce_precision_to_scalar, random_normal_array

class UncertainArray:
    """
    A numerical array with associated elementwise or scalar precision,
    representing a belief/message in a probabilistic inference model.

    This class models complex (or real) multivariate Gaussian beliefs over arrays,
    where each value is represented with a mean (`data`) and an associated precision
    (inverse variance), which can be either a scalar or an array matching the shape of `data`.

    Core functionality includes:
        - Combining beliefs via additive precision (`__mul__`)
        - Subtracting messages to form residuals (`__truediv__`)
        - Damping (convex interpolation of mean and standard deviation)
        - Precision scalarization for approximate factor inference

    Used throughout the factor graph framework to pass messages between
    Wave and Factor nodes during belief propagation.

    Attributes:
        data (np.ndarray): Mean values of the belief (real or complex array).
        precision (Union[float, np.ndarray]): Scalar or elementwise precision.
        dtype (np.dtype): Data type of the array (typically np.complex128).
        shape (tuple): Shape of the data array (excluding batch).
        ndim (int): Number of dimensions of the data array.

    Notes:
        - Internally, precision is stored as `_precision`, which can be a float or ndarray.
        - All arithmetic operations promote precision to elementwise form as needed.
        - If precision is scalar, it is broadcasted for computation but not stored as array.
    """

    def __init__(self, array, dtype=np.complex128, precision=1.0): 
        self.data = np.asarray(array, dtype=dtype)
        self.dtype = self.data.dtype
        self._set_precision_internal(precision)
    
    def is_real(self):
        """Return True if the array has a real-valued dtype (e.g., float32/64)."""
        return np.issubdtype(self.dtype, np.floating)

    def is_complex(self):
        """Return True if the array has a complex-valued dtype (e.g., complex64/128)."""
        return np.issubdtype(self.dtype, np.complexfloating)

    def astype(self, dtype):
        """
        Return a new UncertainArray with data cast to the specified dtype.

        The precision is preserved. A warning is issued if casting from complex
        to real would discard a non-negligible imaginary part.
    
        Args:
            dtype (np.dtype): Target data type (e.g., np.float64, np.complex128).

        Returns:
            UncertainArray: Converted array with same shape and precision.
        """

        if np.issubdtype(dtype, np.floating) and self.is_complex():
            if not np.allclose(self.data.imag, 0):
                import warnings
                warnings.warn("Casting complex array to real will discard imaginary part.")
        return UncertainArray(
            array=self.data.astype(dtype),
            dtype=dtype,
            precision=self._precision
        )

    def _set_precision_internal(self, value):
        """Internal method to set precision as scalar or array."""
        if np.isscalar(value):
            if value <= 0:
                raise ValueError("Precision must be positive.")
            self._precision = float(value)
        else:
            # convert if not already correct dtype
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                arr = value
            else:
                arr = np.asarray(value, dtype=np.float64)
            #check shape
            if arr.shape != self.data.shape:
                raise ValueError("Precision array must match data shape.")
            self._precision = arr

    @property
    def precision(self):
        """Return precision as array (broadcasted if scalar)."""
        if np.isscalar(self._precision):
            return np.full(self.shape, self._precision, dtype=np.float64)
        return self._precision

    @property
    def shape(self):
        """Return the shape of the data array."""
        return self.data.shape
    
    @property
    def ndim(self):
        """Return the number of dimensions of the data array."""
        return self.data.ndim

    @classmethod
    def random(cls, shape, dtype=np.complex128, rng=None, precision=1.0):
        """
        Create a random UncertainArray with Gaussian-distributed complex or real values.

        Args:
            shape (tuple): Shape of the array.
            dtype (np.dtype): Data type (default: np.complex128).
            rng (np.random.Generator or None): Optional RNG for reproducibility.
            precision (float or ndarray): Initial precision (scalar or array).

        Returns:
            UncertainArray: Randomly initialized instance.
        """
        rng = np.random.default_rng() if rng is None else rng
        data = random_normal_array(shape, dtype=dtype, rng=rng)
        return cls(data, dtype=dtype, precision=precision)
    
    @classmethod
    def zeros(cls, shape, dtype=np.complex128, precision=1.0):
        """
        Create an UncertainArray filled with zeros and a specified precision.

        Args:
            shape (tuple): Shape of the array.
            dtype (np.dtype): Data type of the values (default: np.complex128).
            precision (float or ndarray): Initial precision (scalar or array).

        Returns:
            UncertainArray: Zero-initialized instance.
        """
        data = np.zeros(shape, dtype=dtype)
        return cls(data, dtype=dtype, precision=precision)

    
    def __mul__(self, other):
        """
        Combine two UncertainArrays using additive precision rule.

        This operation corresponds to fusing two independent Gaussian beliefs.
        The resulting data is the weighted average, and the precision is summed.

        Returns:
            UncertainArray: The fused belief.
        """
        if not isinstance(other, UncertainArray):
            raise TypeError("Multiplication only supported between UncertainArrays.")
        if self.dtype != other.dtype:
            raise TypeError(f"Dtype mismatch: {self.dtype} vs {other.dtype}")

        d1, d2 = self.data, other.data
        p1, p2 = self._precision, other._precision
        precision_sum = p1 + p2
        result_data = (p1 * d1 + p2 * d2) / precision_sum
        
        return UncertainArray(result_data, dtype=self.dtype, precision=precision_sum)


    def __truediv__(self, other):
        """
        Compute the 'difference' between two UncertainArrays under
        the additive precision model.

        This corresponds to computing the residual message in belief propagation.
        It "removes" the contribution of `other` from `self`.

        Returns:
            UncertainArray: The residual message.
        
        Note:
        The precision subtraction may lead to non-positive values; we clip the result
        to ensure numerical stability and avoid division by zero.
        """
        if not isinstance(other, UncertainArray):
            raise TypeError("Division only supported between UncertainArrays.")
        if self.dtype != other.dtype:
            raise TypeError(f"Dtype mismatch: {self.dtype} vs {other.dtype}")

        d1, d2 = self.data, other.data
        p1, p2 = self._precision, other._precision

        if np.isscalar(p1) and np.isscalar(p2):
            precision_diff = p1 - p2
            precision_safe = max(precision_diff, 1.0)
            result_data = (p1 * d1 - p2 * d2) / precision_safe
        else:
            p1_arr = self.precision
            p2_arr = other.precision
            precision_diff = p1_arr - p2_arr
            precision_safe = np.clip(precision_diff, 1.0, None)
            result_data = (p1_arr * d1 - p2_arr * d2) / precision_safe

        return UncertainArray(result_data, dtype=self.dtype, precision=precision_safe)
        
    def damp_with(self, other, alpha: float) -> "UncertainArray":
        """
        Return a damped version of self, interpolated with another UA instance.

        This method performs convex interpolation of the mean and the standard deviation
        (not precision) using the given damping coefficient alpha.

        Args:
            other (UncertainArray): Target UA to interpolate towards.
            alpha (float): Interpolation factor in [0, 1].

        Returns:
            UncertainArray: Damped instance between self and other.
        """

        if not isinstance(other, UncertainArray):
            raise TypeError("Damping only supported between UncertainArray instances.")
        if self.shape != other.shape:
            raise ValueError("UncertainArrays must have the same shape to apply damping.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Damping coefficient alpha must be between 0 and 1.")

        damped_data = (1 - alpha) * self.data + alpha * other.data

        p1 = self._precision
        p2 = other._precision

        std1 = np.sqrt(1.0 / p1)
        std2 = np.sqrt(1.0 / p2)
        damped_std = (1 - alpha) * std1 + alpha * std2
        damped_precision = 1.0 / (damped_std ** 2)

        return UncertainArray(damped_data, dtype=self.data.dtype, precision=damped_precision)
    
    def as_scalar_precision(self) -> "UncertainArray":
        """
        Return a copy of the current UA with precision replaced by a single scalar value.

        This is used in certain approximate inference scenarios (e.g., in Sparse Prior),
        where a single global precision is needed for efficiency or model consistency.

        Returns:
            UncertainArray: New instance with same data and scalar precision.
        """

        if np.isscalar(self._precision):
            return self
        else:
            scalar_prec = reduce_precision_to_scalar(self._precision)
            return UncertainArray(self.data, dtype=self.data.dtype, precision=scalar_prec)

    def __repr__(self):
        """
        Return a string representation showing the array shape
        and whether precision is scalar or array-valued.
        """
        return f"UA(shape={self.shape}, precision={'scalar' if np.isscalar(self._precision) else 'array'})"

