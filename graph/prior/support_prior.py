from ...core.backend import np
from typing import Optional, Any

from .base import Prior
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import random_normal_array
from ...core.types import PrecisionMode
from ...core.rng_utils import get_rng

class SupportPrior(Prior):
    """
    A structured prior that enforces known support constraints on the latent variable.

    This prior models a variable as:
        - Gaussian CN(0, 1) or N(0, 1) on the support region
        - Deterministically zero (delta function) elsewhere

    Internally, this is implemented via an UncertainArray with a very large precision
    (e.g., 1e10) outside the support, effectively clamping those values to zero.

    Behavior:
        - Precision mode is typically array-based (default), but scalar fallback is supported
        - Forward messages are fixed and reused unless scalar conversion is needed
        - Sampling uses CN(0,1) or N(0,1) on the support, zero elsewhere

    Args:
        support (np().ndarray[bool]): Boolean mask of the same shape as the latent variable.
        dtype (np().dtype): np().float64 or np().complex128 (default: complex).
        precision_mode (PrecisionMode | None): Precision mode to use; defaults to array.
        label (str | None): Optional label for the output wave.

    Attributes:
        support (np().ndarray): Mask indicating active support positions.
        large_value (float): Precision value used to approximate delta constraints.
    """

    def __init__(
        self,
        support: Any, #backend ndarray
        dtype: np().dtype = np().complex128,
        precision_mode: Optional[PrecisionMode] = PrecisionMode.ARRAY,
        label: Optional[str] = None
    ) -> None:

        if support.dtype != bool:
            raise ValueError("Support must be a boolean numpy array.")

        self.support: np().ndarray = support
        self.large_value: float = 1e10

        super().__init__(
            shape=support.shape,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label
        )

        self._fixed_msg_array: UA = self._create_fixed_array(dtype)
    
    def to_backend(self) -> None:
        """
        Convert internal numpy arrays (support mask, fixed messages) to current backend.
        Should be called during Graph.compile() to ensure compatibility with the selected backend.
        """
        if hasattr(self.support, "get") and not hasattr(np(), "cuda"):  
            # numpy backendの場合のみ get() してCPUへ転送
            self.support = self.support.get()
        self.support = np().asarray(self.support, dtype=bool)
        self._fixed_msg_array = self._create_fixed_array(self.dtype)
        self.dtype = self._fixed_msg_array.dtype

    def _create_fixed_array(self, dtype: np().dtype) -> UA:
        self.support = np().asarray(self.support, dtype=bool)  
        mean = np().zeros_like(self.support, dtype=dtype)
        precision = np().where(self.support, 1.0, self.large_value)
        return UA(mean, dtype=dtype, precision=precision)


    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.ARRAY:
            return self._fixed_msg_array
        elif mode == PrecisionMode.SCALAR:
            combined = self._fixed_msg_array * incoming.as_array_precision()
            reduced = combined.as_scalar_precision()
            return reduced / incoming
        else:
            raise RuntimeError("Precision mode not determined for SupportPrior output.")

    def generate_sample(self, rng: Optional[Any]) -> None:
        """
        Generate a sample with N(0,1) or CN(0,1) on support=True, and 0 elsewhere.
        """
        if rng is None:
            rng = get_rng()

        sample = np().zeros(self.support.shape, dtype=self.dtype)
        values = random_normal_array(self.support.shape, dtype=self.dtype, rng=rng)
        sample[self.support] = values[self.support]
        self.output.set_sample(sample)
    
    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return a sample with N(0,1) or CN(0,1) on support=True, and 0 elsewhere.

        Args:
            rng (Optional[Any]): Optional random generator.

        Returns:
            np().ndarray: Sample array matching shape and dtype of the prior.
        """
        if rng is None:
            rng = get_rng()

        sample = np().zeros(self.support.shape, dtype=self.dtype)
        values = random_normal_array(self.support.shape, dtype=self.dtype, rng=rng)
        sample[self.support] = values[self.support]
        return sample

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"SupportPrior(gen={gen}, mode={mode})"
