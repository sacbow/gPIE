from ...core.backend import np, move_array_to_current_backend
from typing import Optional, Any

from .base import Prior
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import random_normal_array
from ...core.types import PrecisionMode, get_real_dtype
from ...core.rng_utils import get_rng


class SupportPrior(Prior):
    """
    A structured prior that enforces known support constraints on the latent variable.

    This prior models a variable as:
        - Gaussian CN(0, 1) or N(0, 1) on the support region
        - Deterministically zero (delta function) elsewhere

    Internally, this is implemented via an UncertainArray with a very large precision
    (e.g., 1e10) outside the support, effectively clamping those values to zero.

    Args:
        support (np().ndarray[bool]):
            Boolean mask of shape `event_shape` or `(batch_size, *event_shape)`.
        dtype (np().dtype):
            Data type (e.g., np().float32 or np().complex64).
        precision_mode (PrecisionMode | None):
            Precision mode to use; defaults to 'array'.
        label (str | None):
            Optional label for the output wave.
    """

    def __init__(
        self,
        support: Any,  # backend ndarray (bool)
        event_shape: tuple[int, ...] = None,
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[PrecisionMode] = PrecisionMode.ARRAY,
        label: Optional[str] = None
    ) -> None:
        """
        Initialize SupportPrior.

        Args:
            support (ndarray[bool]): Boolean mask of shape `event_shape` or `(batch_size, *event_shape)`.
            event_shape (tuple): Shape of each instance (excluding batch dimension).
            batch_size (int): Number of independent instances. Defaults to 1.
            dtype (np().dtype): Data type (e.g., np().complex64).
            precision_mode (PrecisionMode): Defaults to ARRAY.
            label (str): Optional label for the output wave.

        Raises:
            ValueError: If support has invalid shape or dtype.
        """
        if support.dtype != bool:
            raise ValueError("Support mask must be a boolean array.")
        
        #infer event_shape from support if none
        if event_shape is None:
            event_shape = support.shape
        expected_shape = (batch_size,) + event_shape

        # Accept (event_shape,) and broadcast if needed
        if support.shape == event_shape:
            support = np().broadcast_to(support, expected_shape)
        elif support.shape != expected_shape:
            raise ValueError(
                f"Support shape {support.shape} is invalid. Must be {event_shape} or {expected_shape}."
            )

        self.support: np().ndarray = support
        self.large_value: float = get_real_dtype(dtype)(1e10)

        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label
        )

        self._fixed_msg_array: UA = self._create_fixed_array(dtype)


    def _create_fixed_array(self, dtype: np().dtype) -> UA:
        """
        Create a fixed UncertainArray with precision = 1 inside support,
        and large_value outside support.
        """
        real_dtype = get_real_dtype(dtype)
        mean = np().zeros_like(self.support, dtype=dtype)
        precision = np().where(self.support, real_dtype(1.0), self.large_value)
        return UA(mean, dtype=dtype, precision=precision)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the prior message according to precision_mode.
        """
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.ARRAY:
            return self._fixed_msg_array
        elif mode == PrecisionMode.SCALAR:
            combined = self._fixed_msg_array * incoming.as_array_precision()
            reduced = combined.as_scalar_precision()
            return reduced / incoming
        else:
            raise RuntimeError("Precision mode not determined for SupportPrior output.")

    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return a sample with N(0,1) or CN(0,1) on support=True, and 0 elsewhere.

        Returns:
            ndarray: sample of shape (batch_size, *event_shape), dtype = self.dtype
        """
        if rng is None:
            rng = get_rng()

        sample = np().zeros(self.support.shape, dtype=self.dtype)
        values = random_normal_array(self.support.shape, dtype=self.dtype, rng=rng)
        sample[self.support] = values[self.support]
        return sample



    def to_backend(self) -> None:
        """
        Convert internal support mask and cached message to current backend.
        Ensures compatibility with backend-agnostic computation (e.g., NumPy or CuPy).
        """
        self.support = move_array_to_current_backend(self.support, dtype=bool)
        self._fixed_msg_array = self._create_fixed_array(self.dtype)
        self.dtype = self._fixed_msg_array.dtype



    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"SupportPrior(gen={gen}, mode={mode})"
