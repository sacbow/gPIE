from ...core.backend import np, move_array_to_current_backend
from typing import Optional, Tuple, Any

from .base import Prior
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_real_dtype
from ...core.linalg_utils import random_normal_array
from ...core.rng_utils import get_rng

class GaussianPrior(Prior):
    """
    A standard zero-mean Gaussian prior for latent variables.

    This class instantiates a prior distribution of the form:
        - For real dtype:     x ~ N(0, var)
        - For complex dtype:  x ~ CN(0, var)

    The prior defines the origin of a `Wave` in the factor graph, initializing it
    with a fixed variance and zero mean. It supports both scalar and array precision modes.

    Behavior:
        - On the first forward pass, it samples from the prior (if not cached)
        - On later passes, it returns a fixed UncertainArray with the specified precision
        - Sampling and precision are consistent with the specified dtype and mode

    Args:
        var (float): Variance of the Gaussian (must be positive).
        shape (tuple[int, ...]): Shape of the latent variable.
        dtype (np().dtype): Output dtype; supports np().float64 or np().complex128.
        precision_mode (PrecisionMode | None): Scalar or array mode (or inferred).
        label (str | None): Optional name for the associated Wave.

    Attributes:
        var (float): Variance of the prior (used for sampling and precision).
        precision (float): Inverse of variance, used in message construction.
    """
    def __init__(
        self,
        var: float = 1.0,
        event_shape: tuple[int, ...] = (1,),
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None
    ) -> None:
        if var <= 0:
            raise ValueError("Variance must be positive.")
        
        real_dtype = get_real_dtype(dtype)
        self.var: float = real_dtype(var)
        self.precision: float = np().asarray(1.0 / var)

        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label
        )
    
    def to_backend(self):
        move_array_to_current_backend(self.precision)


    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            return UA.zeros(
                event_shape=self.event_shape,
                batch_size=self.batch_size,
                dtype=self.dtype,
                precision=self.precision,
                scalar_precision=True,
            )
        elif mode == PrecisionMode.ARRAY:
            return UA.zeros(
                event_shape=self.event_shape,
                batch_size=self.batch_size,
                dtype=self.dtype,
                precision=self.precision,
                scalar_precision=False,
            )
        else:
            raise RuntimeError("Precision mode not determined for GaussianPrior output.")

    
    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return a sample from the Gaussian prior.

        Returns:
            ndarray: shape = (batch_size, *event_shape), dtype = self.dtype
        """
        if rng is None:
            rng = get_rng()

        shape = (self.batch_size,) + self.event_shape
        return np().sqrt(self.var) * random_normal_array(shape, dtype=self.dtype, rng=rng)



    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"GaussianPrior(gen={gen}, mode={mode}, var={self.var})"
