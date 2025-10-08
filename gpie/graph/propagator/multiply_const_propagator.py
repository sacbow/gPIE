from typing import Union, Optional
from ..wave import Wave
from .base import Propagator
from ...core.uncertain_array import UncertainArray as UA
from ...core.backend import np, move_array_to_current_backend
from ...core.types import (
    PrecisionMode,
    UnaryPropagatorPrecisionMode,
    get_lower_precision_dtype,
    get_real_dtype,
)


class MultiplyConstPropagator(Propagator):
    def __init__(self, const: Union[float, complex, np().ndarray]):
        super().__init__(input_names=("input",))
        self.const = np().asarray(const)
        self.const_dtype = self.const.dtype
        self._init_rng = None

        abs_vals = np().abs(self.const)
        eps = np().array(1e-8, dtype=abs_vals.dtype)
        self.const_safe = self.const.copy()
        self.const_safe[abs_vals < eps] = eps * np().exp(1j * np().angle(self.const_safe[abs_vals < eps]))
        self.inv_amp_sq = 1 / np().abs(self.const_safe) ** 2

    def to_backend(self):
        self.const = move_array_to_current_backend(self.const, dtype=self.const_dtype)
        self.const_safe = move_array_to_current_backend(self.const_safe, dtype=self.const_dtype)
        self.inv_amp_sq = move_array_to_current_backend(self.inv_amp_sq, dtype=get_real_dtype(self.const_dtype))
        self.const_dtype = self.const.dtype

    def set_init_rng(self, rng):
        self._init_rng = rng
    
    def _set_precision_mode(self, mode: Union[str, UnaryPropagatorPrecisionMode]) -> None:
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)

        if mode not in {
            UnaryPropagatorPrecisionMode.ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
        }:
            raise ValueError(f"Unsupported precision mode for MultiplyConstPropagator: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        mode = self.inputs["input"].precision_mode_enum
        if mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)


    def set_precision_mode_backward(self):
        input_mode = self.inputs["input"].precision_mode_enum
        output_mode = self.output.precision_mode_enum
        if output_mode is None or output_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
        elif input_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
        else:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)


    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self.precision_mode_enum is None:
            return None
        elif self.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.SCALAR
        else:
            return PrecisionMode.ARRAY


    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self.precision_mode_enum is None:
            return None
        elif self.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.SCALAR
        else:
            return PrecisionMode.ARRAY

    def _compute_forward(self, input_msg : UA) -> UA:
        dtype = np().result_type(input_msg.dtype, self.const_dtype)
        input_msg = input_msg.astype(dtype)
        const = self.const_safe.astype(dtype)
        mu = input_msg.data * const
        prec = input_msg.precision(raw=True) * self.inv_amp_sq
        return UA(mu, dtype=dtype, precision=prec)


    def _compute_backward(self, output_msg: UA) -> UA:
        dtype = np().result_type(output_msg.dtype, self.const_dtype)
        const = self.const_safe.astype(dtype)
        mu = output_msg.data / const
        prec = output_msg.precision(raw=True) / self.inv_amp_sq
        return UA(mu, dtype=dtype, precision=prec)

    def forward(self):
        input_msg = self.input_messages[self.inputs["input"]]
        if input_msg is None:
            raise RuntimeError("No message to forward")
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            else:
                scalar = self.output._precision_mode == PrecisionMode.SCALAR
                msg = UA.random(
                event_shape=self.output.event_shape,
                batch_size=self.output.batch_size,
                dtype=self.dtype,
                scalar_precision=scalar,
                rng=self._init_rng,
                )
                self.output.receive_message(self, msg)
                return

        msg = self._compute_forward(input_msg)
        if self.output.precision_mode_enum == PrecisionMode.ARRAY:
            self.output.receive_message(self, msg)
        else:
            qy = (msg * self.output_message.as_array_precision()).as_scalar_precision()
            msg_to_send = qy / self.output_message
            self.output.receive_message(self, msg_to_send)
        return

    def backward(self):
        msg = self._compute_backward(self.output_message)
        if self.inputs["input"].precision_mode_enum == PrecisionMode.ARRAY:
            self.inputs["input"].receive_message(self, msg)
        else:
            input_msg = self.input_messages[self.inputs["input"]]
            qx = (msg * input_msg.as_array_precision()).as_scalar_precision()
            msg_to_send = qx / input_msg
            self.inputs["input"].receive_message(self, msg_to_send)

    def get_sample_for_output(self, rng=None):
        x = self.inputs["input"].get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        const = self.const.astype(x.dtype) if self.const_dtype != x.dtype else self.const
        return x * const

    def __matmul__(self, wave: Wave) -> Wave:
        self.dtype = get_lower_precision_dtype(wave.dtype, self.const_dtype)
        self.const = np().asarray(self.const, dtype=self.dtype)
        self.const_dtype = self.const.dtype
        self.const_safe = self.const_safe.astype(self.dtype)
        self.inv_amp_sq = self.inv_amp_sq.astype(get_real_dtype(self.dtype))

        expected_shape = (wave.batch_size, *wave.event_shape)
        try:
            self.const = np().broadcast_to(self.const, expected_shape)
            self.const_safe = np().broadcast_to(self.const_safe, expected_shape)
            self.inv_amp_sq = np().broadcast_to(self.inv_amp_sq, expected_shape)
        except ValueError:
            raise ValueError(
                f"MultiplyConstPropagator: constant shape {self.const.shape} not broadcastable to wave shape {expected_shape}"
            )

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(event_shape=wave.event_shape, batch_size=wave.batch_size, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output


    def __repr__(self) -> str:
        mode = self.precision_mode or "unset"
        shape_str = (
            f"scalar" if np().isscalar(self.const) or self.const.shape == ()
            else f"shape={self.const.shape}"
        )
        return f"MultiplyConst(mode={mode}, {shape_str})"