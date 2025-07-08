import numpy as np
from core.uncertain_array import UncertainArray

class UncertainArrayTensor:
    """
    A batched container for UncertainArray instances, represented as stacked
    arrays along a leading batch dimension.

    This class is primarily used for efficient handling of multiple incoming
    messages (e.g., from child factors) to a single Wave node in belief propagation.
    Each entry corresponds to an individual UncertainArray, and all entries are
    assumed to have the same shape and dtype.

    Attributes:
        data (np.ndarray): Mean values of shape (B, ...), where B is the batch size.
        precision (np.ndarray): Corresponding precision values of shape (B, ...).
        dtype (np.dtype): Data type of the entries (typically np.complex128).
        shape (tuple): Shape of a single UncertainArray (i.e., excluding batch dimension).
        batch_size (int): Number of UncertainArrays in the tensor (i.e., B).
    
    Notes:
        - Precision is always stored as an elementwise array (not scalar).
        - This class is typically not used for standalone inference,
          but as part of message aggregation inside Wave nodes.
    """

    def __init__(self, data, precision, dtype=np.complex128):
        data = np.asarray(data, dtype=dtype)
        precision = np.asarray(precision, dtype=np.float64)

        if precision.shape != data.shape:
            if precision.ndim == 1 and precision.shape[0] == data.shape[0]:
                # Precision is scalar per instance (B,)
                self._scalar_precision = True
            else:
                raise ValueError("Shape mismatch between data and precision.")
        else:
            self._scalar_precision = False

        self.data = data
        self.precision = precision
        self.dtype = dtype
        self.batch_size = data.shape[0]
        self.shape = data.shape[1:]

    @classmethod
    def from_list(cls, ua_list):
        """
        Create an UncertainArrayTensor from a list of UncertainArray instances.

        Args:
            ua_list (List[UncertainArray]): List of UncertainArrays with same shape and dtype.

        Returns:
            UncertainArrayTensor: A stacked tensor combining all given instances.
    
        Raises:
            ValueError: If the list is empty or shapes/dtypes are inconsistent.
        """
        if not ua_list:
            raise ValueError("Empty list provided.")

        data = np.stack([ua.data for ua in ua_list])
        precision = np.stack([ua.precision for ua in ua_list])
        dtype = ua_list[0].dtype

        return cls(data, precision, dtype=dtype)

    def to_list(self):
        """
        Decompose the tensor into a list of UncertainArray instances.

        Returns:
            List[UncertainArray]: Individual components corresponding to each batch entry.
        """
        return [
            UncertainArray(self.data[i], dtype=self.dtype, precision=self.precision[i])
            for i in range(self.batch_size)
        ]

    def combine(self):
        """
        Combine the batch of UncertainArrays using the additive precision rule.
        Supports both scalar and array precision modes.

        Returns:
            UncertainArray: The fused belief across the batch.
        """
        if self._scalar_precision:
            # self.precision shape: (B,)
            # Compute scalar precision sum
            precision_sum = np.sum(self.precision)  # scalar
            # Broadcast for weighted sum
            weights = self.precision[:, np.newaxis]
            weighted_sum = np.sum(weights * self.data, axis=0)
            mean = weighted_sum / precision_sum
            return UncertainArray(mean, dtype=self.dtype, precision=precision_sum)
        else:
            total_precision = np.sum(self.precision, axis=0)
            total_precision_safe = np.clip(total_precision, 1e-12, np.inf)
            weighted_sum = np.sum(self.precision * self.data, axis=0)
            mean = weighted_sum / total_precision_safe
            return UncertainArray(mean, dtype=self.dtype, precision=total_precision_safe)

    def __getitem__(self, idx):
        """
        Access individual UncertainArray by index.
        """
        return UncertainArray(self.data[idx], dtype=self.dtype, precision=self.precision[idx])


    def __len__(self):
        return self.batch_size

    def __repr__(self):
        return f"UncertainArrayTensor(batch_size={self.batch_size}, shape={self.shape})"
