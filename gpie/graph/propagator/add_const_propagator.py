from typing import Union, Optional
from ..wave import Wave
from .base import Propagator
from ...core.backend import np, move_array_to_current_backend
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, get_complex_dtype, get_lower_precision_dtype


class AddConstPropagator(Propagator):
    """
    Adds a fixed constant (scalar or array) to the input wave.
    Supports scalar and batch-wise broadcasting, with dtype safety.
    """

    def __init__(self, const: Union[float, complex, np().ndarray]):
        super().__init__(input_names=("input",))
        self.const = np().asarray(const)
        self.const_dtype = self.const.dtype
        self._init_rng = None

    def to_backend(self):
        self.const = move_array_to_current_backend(self.const, dtype=self.const_dtype)
        self.const_dtype = self.const.dtype

    def _set_precision_mode(self, mode: Union[str, PrecisionMode]) -> None:
        if isinstance(mode, str):
            mode = PrecisionMode(mode)
        if mode.value not in ("scalar", "array"):
            raise ValueError(f"Invalid precision mode for AddConstPropagator: {mode}")
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(f"Precision mode conflict: existing={self._precision_mode}, new={mode}")
        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        return self._precision_mode

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        return self._precision_mode

    def set_precision_mode_forward(self):
        mode = self.inputs["input"].precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode)

    def set_precision_mode_backward(self):
        mode = self.output.precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode)

    def _compute_forward(self, inputs: dict[str, UA], block = None) -> UA:
        x = inputs["input"]
        target_dtype = np().result_type(x.dtype, self.const_dtype)
        if x.dtype != target_dtype:
            x = x.astype(target_dtype)

        const = self.const.astype(target_dtype) if self.const_dtype != target_dtype else self.const
        return UA(x.data + const, dtype=target_dtype, precision=x.precision(raw=True))

    def _compute_backward(self, output_msg: UA, exclude: str, block = None) -> UA:
        const = self.const.astype(output_msg.dtype) if output_msg.dtype != self.const_dtype else self.const
        return UA(output_msg.data - const, dtype=output_msg.dtype, precision=output_msg.precision(raw=True))

    def get_sample_for_output(self, rng=None):
        x_sample = self.inputs["input"].get_sample()
        if x_sample is None:
            raise RuntimeError("Input sample not set.")
        const = self.const.astype(x_sample.dtype) if self.const_dtype != x_sample.dtype else self.const
        return x_sample + const

    def __matmul__(self, wave: Wave) -> Wave:
        self.dtype = get_lower_precision_dtype(wave.dtype, self.const_dtype)

        if self.const.dtype != self.dtype:
            self.const = np().asarray(self.const, dtype=self.dtype)
            self.const_dtype = self.const.dtype

        target_shape = (wave.batch_size, *wave.event_shape)

        try:
            self.const = np().broadcast_to(self.const, target_shape)
        except ValueError:
            raise ValueError(f"Const shape {self.const.shape} not broadcastable to wave shape {wave.event_shape}")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(event_shape=wave.event_shape, batch_size=wave.batch_size, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output


    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"AddConst(gen={gen}, mode={self.precision_mode})"
