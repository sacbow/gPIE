from ...core.backend import np, move_array_to_current_backend
from typing import Optional, Tuple, Any, Union

from .base import Prior
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_real_dtype
from ...core.linalg_utils import random_normal_array
from ...core.rng_utils import get_rng


class GaussianPrior(Prior):
    """
    A Gaussian prior for latent variables with optional non-zero mean.

    This class defines a prior distribution of the form:
        - For real dtype:     x ~ N(mean, var)
        - For complex dtype:  x ~ CN(mean, var)

    Behavior:
        - On the first forward pass, it samples from N(mean, var)
        - On later passes, it returns an UncertainArray with the specified precision
        - `mean` may be a scalar or an array matching `event_shape`

    Args:
        mean (float | np.ndarray | None): Mean of the Gaussian prior (default: 0).
        var (float): Variance of the Gaussian (must be positive).
        event_shape (tuple[int, ...]): Shape of the latent variable.
        batch_size (int): Number of independent samples.
        dtype (np().dtype): Data type (e.g., np().float32 or np().complex64).
        precision_mode (PrecisionMode | None): Scalar or array mode.
        label (str | None): Optional name for the associated Wave.
    """

    def __init__(
        self,
        mean: Optional[Union[float, np().ndarray]] = 0.0,
        var: float = 1.0,
        event_shape: tuple[int, ...] = (1,),
        *,
        batch_size: int = 1,
        dtype: np().dtype = np().complex64,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None,
    ) -> None:
        if var <= 0:
            raise ValueError("Variance must be positive.")

        real_dtype = get_real_dtype(dtype)
        self.var: float = real_dtype(var)
        self.precision: float = np().asarray(1.0 / var)

        # --- mean handling ---
        if mean is None:
            self.mean = np().zeros(event_shape, dtype=dtype)
        elif np().isscalar(mean):
            self.mean = np().full(event_shape, mean, dtype=dtype)
        else:
            arr = np().asarray(mean, dtype=dtype)
            if arr.shape != event_shape:
                raise ValueError(
                    f"Mean shape mismatch: expected {event_shape}, got {arr.shape}"
                )
            self.mean = arr

        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label,
        )

    def to_backend(self):
        """Move internal arrays to current backend."""
        self.mean = move_array_to_current_backend(self.mean, dtype=self.dtype)
        self.precision = move_array_to_current_backend(self.precision)

    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode_enum
        scalar_precision = mode == PrecisionMode.SCALAR

        shape = (self.batch_size,) + self.event_shape

        if scalar_precision:
            precision = self.precision
        else:
            precision = np().full(shape, self.precision, dtype=get_real_dtype(self.dtype))

        mean = np().broadcast_to(self.mean, shape)
        return UA(mean, dtype=self.dtype, precision=precision, batched=True)


    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return a sample from the Gaussian prior: mean + sqrt(var) * ε.
        """
        if rng is None:
            rng = get_rng()

        shape = (self.batch_size,) + self.event_shape
        noise = random_normal_array(shape, dtype=self.dtype, rng=rng)
        return self.mean + np().sqrt(self.var) * noise

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        mean_desc = (
            "scalar" if np().isscalar(self.mean) else f"array(shape={self.mean.shape})"
        )
        return (
            f"GaussianPrior(gen={gen}, mode={mode}, mean={mean_desc}, var={self.var})"
        )
