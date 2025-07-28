from __future__ import annotations

from typing import List, Union, Literal
from numpy.typing import NDArray
from .backend import np
from .types import ArrayLike, PrecisionMode, Precision, get_real_dtype
from .linalg_utils import reduce_precision_to_scalar, random_normal_array

from .uncertain_array import UncertainArray
from .types import PrecisionMode

class UncertainArrayTensor:
    """
    A batched container for UncertainArray instances, stacked along a leading dimension.

    This class represents a collection of Gaussian-distributed beliefs/messages
    with the same shape and dtype, enabling efficient batch operations during
    expectation propagation (EP), particularly within Wave nodes.

    The container distinguishes two precision modes:
        - 'scalar': Each UA has a single scalar precision (stored as shape (B,))
        - 'array': Each UA has full elementwise precision (stored as shape (B, ...))

    This distinction impacts how the `combine()` operation fuses messages, and
    is automatically inferred based on the shape of the input precision tensor.

    Typical use cases:
    - Representing incoming messages from multiple child factors in a Wave
    - Supporting batch belief fusion via `combine()` in message-passing inference
    - Enabling efficient vectorized operations on groups of UncertainArray

    Attributes:
        data (np().ndarray): Mean values of shape (B, ...), where B is the batch size.
        precision (np().ndarray): Precision values of shape (B,) or (B, ...).
        dtype (np().dtype): Data type of the belief means.
        shape (tuple): Shape of an individual UncertainArray (i.e., excluding batch dim).
        batch_size (int): Number of UncertainArrays in the batch.
        precision_mode (str): Either 'scalar' or 'array'.
    """


    def __init__(
        self,
        data: NDArray,
        precision: Union[NDArray, NDArray[np().float64]],
        dtype: np().dtype = np().complex128
    ) -> None:
        """
        Initialize an UncertainArrayTensor from stacked data and precision arrays.

        Args:
            data: Mean values of shape (B, ...), stacked across batch dimension.
            precision: Precision values. Shape (B,) implies scalar precision,
                   shape (B, ...) implies array precision.
            dtype: Data type of the mean values (default: np().complex128).

        Raises:
            ValueError: If shape mismatch between data and precision.
        """
        data = np().asarray(data, dtype=dtype)
        precision = np().asarray(precision, dtype=get_real_dtype(dtype))

        if precision.shape != data.shape:
            if precision.ndim == 1 and precision.shape[0] == data.shape[0]:
                # Precision is scalar per instance (B,)
                self._scalar_precision = True
                precision = np().reshape(precision, (precision.shape[0],) + (1,) * (data.ndim - 1))
            else:
                raise ValueError("Shape mismatch between data and precision.")
        else:
            self._scalar_precision = False

        self.data: NDArray = data
        self.precision: NDArray = precision
        self.dtype: np().dtype = dtype
        self.batch_size: int = data.shape[0]
        self.shape: tuple[int, ...] = data.shape[1:]
    
    def to_backend(self) -> None:
        """
        Move data and precision to current backend (NumPy or CuPy).
        Updates dtype based on backend-casted array.
        """
        self.data = np().asarray(self.data)
        self.precision = np().asarray(self.precision, dtype=get_real_dtype(self.dtype))
        self.dtype = self.data.dtype  # Update dtype after backend switch

    
    @property
    def precision_mode(self) -> PrecisionMode:
        """
        Return the precision mode of the tensor: either 'scalar' or 'array'.

        - 'scalar': Precision is of shape (B,)
        - 'array':  Precision is of shape (B, ...) matching data

        This affects how operations like `combine()` are computed.
        """

        return PrecisionMode.SCALAR if self._scalar_precision else PrecisionMode.ARRAY
    
    def assert_compatible(self, ua: UncertainArray, idx: int | None = None, context: str = "") -> None:
        """
        Assert that a given UncertainArray is compatible with this tensor.

        Checks that dtype, shape, and precision_mode all match. Used in __setitem__ and other
        batch update contexts (e.g., message passing in Wave nodes).

        Args:
            ua: The UncertainArray to compare against.
            idx: Optional index (for error messages).
            context: Optional context string (e.g., "__setitem__").

        Raises:
            ValueError: If shape or precision_mode mismatch.
            TypeError: If dtype mismatch.
        """
        prefix = f"[{context}] " if context else ""
        suffix = f" at index {idx}" if idx is not None else ""

        if ua.precision_mode != self.precision_mode:
            raise ValueError(
                f"{prefix}Precision mode mismatch{suffix}: "
                f"UAT is '{self.precision_mode}', UA is '{ua.precision_mode}'"
            )
        if ua.shape != self.shape:
            raise ValueError(
                f"{prefix}Shape mismatch{suffix}: "
                f"UAT expects {self.shape}, got {ua.shape}"
            )
        if ua.dtype != self.dtype:
            raise TypeError(
                f"{prefix}Dtype mismatch{suffix}: "
                f"UAT expects {self.dtype}, got {ua.dtype}"
            )
    
    def __setitem__(self, idx: int, ua: UncertainArray) -> None:
        """
        Replace the UncertainArray at index `idx` with `ua`.

        Performs compatibility checks on dtype, shape, and precision_mode
        to prevent silent runtime broadcasting or shape mismatch errors.

        Args:
            idx: Index of the batch item to replace.
            ua: The UncertainArray instance to insert.

        Raises:
            ValueError or TypeError if compatibility check fails.
        """
        self.assert_compatible(ua, idx=idx, context="__setitem__")
        self.data[idx] = ua.data
        self.precision[idx] = ua.precision(raw=True)

    @classmethod
    def from_list(cls, ua_list: list[UncertainArray]) -> UncertainArrayTensor:
        """
        Create a UncertainArrayTensor from a list of UncertainArrays.

        All instances must have the same shape, dtype, and precision_mode.

        This method is primarily provided for testing, debugging, or bridging
        legacy code. Internally, tensorization is preferred for performance.

        Args:
            ua_list: List of UncertainArray instances.

        Returns:
            UncertainArrayTensor: Batched representation of the list.

        Raises:
            ValueError / TypeError: If any array is incompatible.
        
        Note:
        This method is primarily intended for testing and debugging.
        In production pipelines, prefer constructing tensors directly for performance.
        """

        if not ua_list:
            raise ValueError("Empty list provided.")

        ref = ua_list[0]
        data = [ua.data for ua in ua_list]
        prec = [np().asarray(ua.precision(raw=True), dtype=get_real_dtype(ref.dtype)) for ua in ua_list]

        tmp = cls(np().stack(data), np().stack(prec), dtype=ref.dtype)

        for i, ua in enumerate(ua_list):
            tmp.assert_compatible(ua, idx=i, context="from_list")

        return tmp


    def to_list(self) -> list[UncertainArray]:
        """
        Decompose the tensor into a list of UncertainArray instances.

        Returns:
            List[UncertainArray]: Individual components corresponding to each batch entry.
        """
        if self._scalar_precision:
            # precision[i] is shape (1, ..., 1), so convert to scalar
            return [
                UncertainArray(self.data[i], dtype=self.dtype, precision=float(self.precision[i].reshape(-1)[0]))
                for i in range(self.batch_size)
            ]
        else:
            return [
                UncertainArray(self.data[i], dtype=self.dtype, precision=self.precision[i])
                for i in range(self.batch_size)
            ]

    def combine(self) -> UncertainArray:
        """
        Fuse the batch of UncertainArrays into a single belief via precision-weighted averaging.

        This operation corresponds to combining multiple messages in belief propagation.

        - In scalar mode: weights each UA by its scalar precision, outputs global mean and precision.
        - In array mode: performs elementwise fusion over each position.

        Returns:
            UncertainArray: Fused belief with same shape and dtype as individual UAs.
        """

        if self._scalar_precision:
            # precision: shape (B, 1, ..., 1)
            precision_sum = np().sum(self.precision, axis=0)  # shape: (1, ..., 1)
            weighted_sum = np().sum(self.precision * self.data, axis=0)
            mean = weighted_sum / precision_sum
            scalar_precision = float(precision_sum.reshape(-1)[0])
            return UncertainArray(mean, dtype=self.dtype, precision=scalar_precision)


        else:
            total_precision = np().sum(self.precision, axis=0)
            weighted_sum = np().sum(self.precision * self.data, axis=0)
            mean = weighted_sum / total_precision
            return UncertainArray(mean, dtype=self.dtype, precision=total_precision)

    def __getitem__(self, idx: int) -> UncertainArray:
        """
        Access individual UncertainArray by index.

        Args:
            idx (int): Index into the batch dimension.

        Returns:
            UncertainArray: The UncertainArray at the given batch index.
        """
        prec = self.precision[idx]
        if self._scalar_precision:
            prec = float(prec.reshape(-1)[0])  # flattenしてスカラー化
        return UncertainArray(self.data[idx], dtype=self.dtype, precision=prec)

    def __len__(self) -> int:
        """Return the number of UncertainArrays in the tensor (batch size)."""
        return self.batch_size

    def __repr__(self) -> str:
        return f"UncertainArrayTensor(batch_size={self.batch_size}, shape={self.shape})"
