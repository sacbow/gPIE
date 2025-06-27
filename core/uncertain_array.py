import numpy as np
from .linalg_utils import complex_normal_random_array, reduce_precision_to_scalar

class UncertainArray:
    def __init__(self, array, dtype=np.complex128, precision=1.0):
        # Convert input to ndarray with given dtype
        arr = np.asarray(array, dtype=dtype)
        self.data = arr
        # Store precision (can be scalar or array)
        self._set_precision_internal(precision)

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


    def set_precision(self, value):
        """
        Public method to update precision.
        Accepts either a positive scalar or a positive array with same shape as data.
        """
        self._set_precision_internal(value)

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
    def random(cls, shape, dtype=np.complex128, precision=1.0, seed=None):
        data = complex_normal_random_array(shape, dtype=dtype, seed=seed)
        return cls(data, dtype=dtype, precision=precision)

    
    @classmethod
    def zeros(cls, shape, dtype=np.complex128, precision=1.0):
        """
        Create an UncertainArray with zero-initialized complex data and default precision.
        """
        data = np.zeros(shape, dtype=dtype)
        return cls(data, dtype=dtype, precision=precision)

    
    def __mul__(self, other):
        """Combine two UncertainArrays using additive precision rule."""
        if not isinstance(other, UncertainArray):
            raise TypeError("Multiplication only supported between UncertainArrays.")
        p1 = self._precision
        p2 = other._precision
        d1 = self.data
        d2 = other.data
        if np.isscalar(p1) and np.isscalar(p2):
            precision_sum = p1 + p2
            result_data = (p1 * d1 + p2 * d2) / precision_sum
            return UncertainArray(result_data, precision=precision_sum)
        p1_arr = self.precision
        p2_arr = other.precision
        precision_sum = p1_arr + p2_arr
        result_data = (p1_arr * d1 + p2_arr * d2) / precision_sum
        return UncertainArray(result_data, precision=precision_sum)

    
    def __truediv__(self, other):
        """
        Compute the 'difference' between two UncertainArrays under
        the additive precision model. Used in contexts such as belief / input
        to compute messages to factors.
        """
        if not isinstance(other, UncertainArray):
            raise TypeError("Division only supported between UncertainArrays.")

        p1 = self._precision
        p2 = other._precision
        d1 = self.data
        d2 = other.data

        if np.isscalar(p1) and np.isscalar(p2):
            precision_diff = p1 - p2
            precision_safe = max(precision_diff, 1.0)
            result_data = (p1 * d1 - p2 * d2) / precision_safe
            return UncertainArray(result_data, precision=precision_safe)

        p1_arr = self.precision
        p2_arr = other.precision
        precision_diff = p1_arr - p2_arr
        precision_safe = np.clip(precision_diff, 1.0, None)
        result_data = (p1_arr * d1 - p2_arr * d2) / precision_safe
        return UncertainArray(result_data, precision=precision_safe)

    
    @classmethod
    def combine(cls, ua_list):
        """
        Efficiently combine multiple UncertainArrays using additive precision model.
        Assumes all arrays have the same shape.
        """
        if not ua_list:
            raise ValueError("UncertainArray.combine() received an empty list.")

        shape = ua_list[0].shape
        for ua in ua_list:
            if ua.shape != shape:
                raise ValueError("All UncertainArrays must have the same shape.")

        all_scalar = all(np.isscalar(ua._precision) for ua in ua_list)

        datas = np.stack([ua.data for ua in ua_list], axis=0)

        if all_scalar:
            precisions_scalar = np.array([ua._precision for ua in ua_list])
            precision_sum = np.sum(precisions_scalar)
            weighted_sum = np.tensordot(precisions_scalar, datas, axes=1)
            result_data = weighted_sum / precision_sum
            return cls(result_data, precision=precision_sum)

        precisions = np.stack([ua.precision for ua in ua_list], axis=0)
        precision_sum = np.sum(precisions, axis=0)
        weighted_sum = np.sum(precisions * datas, axis=0)
        result_data = weighted_sum / precision_sum
        return cls(result_data, precision=precision_sum)
    
    def damp_with(self, other, alpha: float) -> "UncertainArray":
        """
        Return a damped version of self, interpolated with another UA instance.
        Damping is applied to the mean and standard deviation, not precision.

        Args:
            other (UncertainArray): The target array to damp toward.
            alpha (float): Damping coefficient. 0 = no damping (return self), 1 = full damping (return other).

        Returns:
            UncertainArray: Damped result.
        """
        if not isinstance(other, UncertainArray):
            raise TypeError("Damping only supported between UncertainArray instances.")
        if self.shape != other.shape:
            raise ValueError("UncertainArrays must have the same shape to apply damping.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Damping coefficient alpha must be between 0 and 1.")

        # Damped mean (complex-valued)
        damped_data = (1 - alpha) * self.data + alpha * other.data

        # Damped standard deviation and precision using NumPy broadcasting
        std1 = np.sqrt(1.0 / self.precision)
        std2 = np.sqrt(1.0 / other.precision)
        damped_std = (1 - alpha) * std1 + alpha * std2
        damped_precision = 1.0 / (damped_std ** 2)

        return UncertainArray(damped_data, dtype=self.data.dtype, precision=damped_precision)


    def to_scalar_precision(self):
        """
        Reduce precision (scalar or array) to a single scalar value.
        If already scalar, returns it directly.
        Otherwise uses harmonic mean of variances via reduce_precision_to_scalar.
        """
        if np.isscalar(self._precision):
            return self._precision
        return reduce_precision_to_scalar(self._precision)
    
    def as_scalar_precision(self) -> "UncertainArray":
        """
        Return a copy of the current UA with precision replaced by scalar equivalent.
        """
        scalar_prec = self.to_scalar_precision()
        return UncertainArray(self.data, dtype=self.data.dtype, precision=scalar_prec)

    def __repr__(self):
        return f"UncertainArray(shape={self.shape}, precision={'scalar' if np.isscalar(self._precision) else 'array'})"
