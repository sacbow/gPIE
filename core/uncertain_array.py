import numpy as np
from .linalg_utils import complex_normal_random_array

class UncertainArray:
    def __init__(self, array, dtype=np.complex128, precision=1.0):
        # Convert input to ndarray with given dtype
        arr = np.array(array, dtype=dtype)
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
            arr = np.array(value, dtype=np.float64)
            if arr.shape != self.data.shape:
                raise ValueError("Precision array must match data shape.")
            if not np.all(arr > 0):
                raise ValueError("All precision values must be positive.")
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
            return np.full(self.data.shape, self._precision, dtype=np.float64)
        return self._precision

    @property
    def shape(self):
        """Return the shape of the data array."""
        return self.data.shape

    @classmethod
    def random(cls, shape, dtype=np.complex128, precision=1.0):
        """
        Create an UncertainArray with random complex data and default precision.
        """
        data = complex_normal_random_array(shape, dtype)
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

        shape = ua_list[0].data.shape
        for ua in ua_list:
            if ua.data.shape != shape:
                raise ValueError("All UncertainArrays must have the same shape.")

        all_scalar = all(np.isscalar(ua._precision) for ua in ua_list)

        datas = np.array([ua.data for ua in ua_list])

        if all_scalar:
            precisions_scalar = np.array([ua._precision for ua in ua_list])
            precision_sum = np.sum(precisions_scalar)
            weighted_sum = np.tensordot(precisions_scalar, datas, axes=1)
            result_data = weighted_sum / precision_sum
            return cls(result_data, precision=precision_sum)

        precisions = np.array([ua.precision for ua in ua_list])
        precision_sum = np.sum(precisions, axis=0)
        weighted_sum = np.sum(precisions * datas, axis=0)
        result_data = weighted_sum / precision_sum
        return cls(result_data, precision=precision_sum)
    
    def to_scalar_precision(self):
        """
        Reduce precision array to an equivalent scalar precision.
        Uses the harmonic mean of per-element variances.
        """
        if np.isscalar(self._precision):
            return self._precision

        precision_array = self._precision
        scalar_precision = 1.0 / np.mean(1.0 / precision_array)
        return scalar_precision




