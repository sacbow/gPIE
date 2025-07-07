import numpy as np
from core.uncertain_array import UncertainArray

class UncertainArrayTensor:
    """
    Represents a tensor of UncertainArray objects stacked along a new axis.
    This is intended for efficient multi-branch message aggregation in Wave nodes.

    Attributes:
        data: ndarray of shape (B, ...) where B is the number of branches.
        precision: scalar or ndarray of shape (B, ...) depending on precision mode.
        dtype: data type (e.g., np.complex128)
        shape: shape of each individual array (excluding batch dimension)
        batch_size: number of UncertainArrays stacked
    """

    def __init__(self, data, precision, dtype=np.complex128):
        if data.ndim < 1:
            raise ValueError("data must have at least 1 dimension (batch dimension required).")
        if np.iscomplexobj(data) and dtype not in (np.complex64, np.complex128):
            raise ValueError("dtype must be a complex type.")
        
        self.data = np.asarray(data, dtype=dtype)
        self.precision = np.asarray(precision, dtype=np.float64)

        if self.data.shape != self.precision.shape:
            raise ValueError("Shape mismatch between data and precision.")

        self.dtype = dtype
        self.batch_size = self.data.shape[0]
        self.shape = self.data.shape[1:]

    @classmethod
    def from_list(cls, ua_list):
        """
        Construct from list of UncertainArray instances.
        """
        if not ua_list:
            raise ValueError("Empty list provided.")

        data = np.stack([ua.data for ua in ua_list])
        precision = np.stack([ua.precision for ua in ua_list])
        dtype = ua_list[0].dtype

        return cls(data, precision, dtype=dtype)

    def to_list(self):
        """
        Convert to list of UncertainArray instances.
        """
        return [
            UncertainArray(self.data[i], dtype=self.dtype, precision=self.precision[i])
            for i in range(self.batch_size)
        ]

    def combine(self):
        """
        Vectorized combination of batch UncertainArrays.
        Returns a single UncertainArray.
        """
        total_precision = np.sum(self.precision, axis=0)
        # Avoid divide-by-zero
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
