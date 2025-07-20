import numpy as np
from typing import Optional, Tuple

from .base import Prior
from core.uncertain_array import UncertainArray as UA
from core.types import PrecisionMode
from core.linalg_utils import random_normal_array


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
        dtype (np.dtype): Output dtype; supports np.float64 or np.complex128.
        precision_mode (PrecisionMode | None): Scalar or array mode (or inferred).
        label (str | None): Optional name for the associated Wave.

    Attributes:
        var (float): Variance of the prior (used for sampling and precision).
        precision (float): Inverse of variance, used in message construction.
    """
    def __init__(
        self,
        var: float = 1.0,
        shape: Tuple[int, ...] = (1,),
        dtype: np.dtype = np.complex128,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None
    ) -> None:

        if var <= 0:
            raise ValueError("Variance must be positive.")
        
        self.var: float = var
        self.precision: float = 1.0 / var

        super().__init__(shape=shape, dtype=dtype, precision_mode=precision_mode, label=label)

    def _compute_message(self, incoming: UA) -> UA:
        """
        Return the fixed-precision zero-mean Gaussian prior as UncertainArray.
        """
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            return UA.zeros(self.shape, dtype=self.dtype, precision=self.precision, scalar_precision=True)
        elif mode == PrecisionMode.ARRAY:
            return UA.zeros(self.shape, dtype=self.dtype, precision=self.precision, scalar_precision=False)
        else:
            raise RuntimeError("Precision mode not determined for GaussianPrior output.")

    def generate_sample(self, rng: Optional[np.random.Generator]) -> None:
        """
        Generate a sample from N(0, var) or CN(0, var) depending on dtype.
        """
        if rng is None:
            rng = np.random.default_rng()

        sample = np.sqrt(self.var) * random_normal_array(self.shape, dtype=self.dtype, rng=rng)
        self.output.set_sample(sample)

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"GaussianPrior(gen={gen}, mode={mode}, var={self.var})"
