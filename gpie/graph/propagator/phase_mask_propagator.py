from typing import Optional
from .base import Propagator
from ...core.backend import np, move_array_to_current_backend
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_complex_dtype


class PhaseMaskPropagator(Propagator):
    """
    Elementwise phase modulation propagator with a fixed unit-magnitude complex mask.

    This propagator applies a componentwise complex multiplication:
        y = x * phase_mask

    where `phase_mask` is a complex-valued array with |phase_mask[i]| = 1.

    Supports batched UncertainArray and arbitrary shape of phase_mask,
    as long as it can be broadcast to (batch_size, *event_shape).
    """

    def __init__(self, phase_mask, dtype=np().complex64):
        super().__init__(input_names=("input",), dtype=dtype)

        if not np().allclose(np().abs(phase_mask), 1.0):
            raise ValueError("Phase mask must have unit magnitude.")

        self.phase_mask = np().asarray(phase_mask)
        self._original_mask = phase_mask  # for possible debug/reference
        self.shape = self.phase_mask.shape  # tentative; true check at __matmul__

    def to_backend(self):
        self.phase_mask = move_array_to_current_backend(self.phase_mask, dtype=self.dtype)
        self.dtype = self.phase_mask.dtype

    def _set_precision_mode(self, mode: str | PrecisionMode) -> None:
        if isinstance(mode, str):
            mode = PrecisionMode(mode)

        if mode.value not in ("scalar", "array"):
            raise ValueError(f"Invalid precision mode for PhaseMaskPropagator: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for PhaseMaskPropagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        return self._precision_mode

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        return self._precision_mode

    def set_precision_mode_forward(self):
        mode = self.inputs["input"].precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode.value)

    def set_precision_mode_backward(self):
        mode = self.output.precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode.value)

    def _compute_forward(self, incoming: dict[str, UA]) -> UA:
        ua = incoming["input"]
        data = ua.data.astype(self.dtype, copy=False)
        result = data * self.phase_mask
        return UA(result, dtype=self.dtype, precision=ua.precision(raw=True))

    def _compute_backward(self, outgoing: UA, exclude: str = None) -> UA:
        data = outgoing.data.astype(self.dtype, copy=False)
        result = data / self.phase_mask
        return UA(result, dtype=self.dtype, precision=outgoing.precision(raw=True))

    def __matmul__(self, wave: Wave) -> Wave:
        # Check broadcast compatibility
        expected_shape = (wave.batch_size,) + wave.event_shape
        try:
            self.phase_mask = np().broadcast_to(self.phase_mask, expected_shape)
        except Exception:
            raise ValueError(
                f"Phase mask of shape {self.phase_mask.shape} is not broadcastable to input Wave shape {expected_shape}"
            )

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = get_complex_dtype(wave.dtype)
        self.phase_mask = self.phase_mask.astype(self.dtype)

        out_wave = Wave(
            event_shape=wave.event_shape,
            batch_size=wave.batch_size,
            dtype=self.dtype,
            precision_mode=wave.precision_mode_enum,
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def get_sample_for_output(self, rng):
        x = self.inputs["input"].get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        x = x.astype(self.dtype, copy=False)
        return x * self.phase_mask

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PhaseMaskProp(gen={gen}, mode={self.precision_mode})"
