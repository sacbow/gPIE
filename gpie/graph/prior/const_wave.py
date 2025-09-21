from ...core.backend import np, move_array_to_current_backend
from typing import Optional, Any

from .base import Prior
from ...core.linalg_utils import random_normal_array
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_real_dtype


class ConstWave(Prior):
    def __init__(
        self,
        data: Any,
        *,
        event_shape: Optional[tuple[int, ...]] = None,
        batch_size: int = 1,
        dtype: Optional[np().dtype] = None,
        large_value: float = 1e8,
        precision_mode: Optional[PrecisionMode] = None,
        label: Optional[str] = None
    ) -> None:
        """
        Deterministic prior with fixed data and high precision (approximate delta function).

        Args:
            data: Fixed mean values for the wave.
            event_shape: Shape of each atomic variable (excluding batch).
            batch_size: Number of batch instances. Default: 1.
            dtype: Optional override for data type.
            large_value: Precision to simulate deterministic behavior.
            precision_mode: "scalar", "array", or None.
            label: Optional wave label.
        """
        # Normalize data
        arr = np().asarray(data)
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        elif dtype is None:
            dtype = arr.dtype

        if event_shape is None:
            # Try to infer event_shape from data
            if batch_size > 1:
                event_shape = arr.shape[1:]
            else:
                event_shape = arr.shape
        expected_shape = (batch_size,) + event_shape

        if arr.shape == expected_shape:
            pass  # OK
        elif arr.shape == event_shape:
            arr = np().broadcast_to(arr, expected_shape)
        else:
            raise ValueError(f"Input data shape {arr.shape} is incompatible with batch_size={batch_size}, event_shape={event_shape}")

        self._data = arr
        self.large_value = np().array(large_value, get_real_dtype(dtype))

        super().__init__(
            event_shape=event_shape,
            batch_size=batch_size,
            dtype=dtype,
            precision_mode=precision_mode,
            label=label
        )

    def to_backend(self) -> None:
        self._data = move_array_to_current_backend(self._data)
        self.dtype = self._data.dtype

    def _compute_message(self, incoming: UA) -> UA:
        mode = self.output.precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            return UA(self._data, dtype=self.dtype, precision=self.large_value)
        elif mode == PrecisionMode.ARRAY:
            prec_array = np().full(self._data.shape, self.large_value, dtype=get_real_dtype(self.dtype))
            return UA(self._data, dtype=self.dtype, precision=prec_array)
        else:
            raise RuntimeError("Precision mode not determined for ConstWave output.")

    def get_sample_for_output(self, rng: Optional[Any] = None) -> np().ndarray:
        """
        Return the fixed sample stored in this constant prior, with optional small noise.

        Args:
            rng (Optional[Any]): Optional RNG.

        Returns:
            np().ndarray: Sampled array with deterministic mean.
        """
        noise = np().sqrt(1.0 / self.large_value) * random_normal_array(
            shape=(self.batch_size,) + self.event_shape,
            dtype=self.dtype,
            rng=rng
        )
        return self._data + noise

    def __repr__(self) -> str:
        gen = self._generation if self._generation is not None else "-"
        mode = self.precision_mode or "None"
        return f"ConstWave(gen={gen}, mode={mode})"
