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
        self.inv_amp_sq = 1.0 / np().abs(self.const_safe) ** 2
        self.inv_amp_sq_scalar = 1.0 / np().mean(np().abs(self.const_safe) ** 2)

    def _set_precision_mode(self, mode: Union[str, UnaryPropagatorPrecisionMode]) -> None:
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)
        if mode == UnaryPropagatorPrecisionMode.SCALAR:
            raise ValueError("MultiplyConstPropagator does not support SCALAR precision mode.")
        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for MultiplyConstPropagator: existing='{self._precision_mode}', requested='{mode}'"
            )
        self._precision_mode = mode

    def to_backend(self):
        self.const = move_array_to_current_backend(self.const, dtype=self.const_dtype)
        self.const_safe = move_array_to_current_backend(self.const_safe, dtype=self.const_dtype)
        self.inv_amp_sq = move_array_to_current_backend(self.inv_amp_sq, dtype=get_real_dtype(self.const_dtype))
        self.inv_amp_sq_scalar = move_array_to_current_backend(
            self.inv_amp_sq_scalar, dtype=get_real_dtype(self.const_dtype)
        )
        self.const_dtype = self.const.dtype

    def set_init_rng(self, rng):
        self._init_rng = rng

    def set_precision_mode_forward(self):
        x_mode = self.inputs["input"].precision_mode_enum
        if x_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    def set_precision_mode_backward(self):
        y_mode = self.output.precision_mode_enum
        x_mode = self.inputs["input"].precision_mode_enum
        if y_mode == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
        elif y_mode == PrecisionMode.ARRAY:
            if x_mode == PrecisionMode.SCALAR:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
            else:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        else:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode in (
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
            UnaryPropagatorPrecisionMode.SCALAR,
        ):
            return PrecisionMode.SCALAR
        return PrecisionMode.ARRAY

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode in (
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR,
        ):
            return PrecisionMode.SCALAR
        return PrecisionMode.ARRAY

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        x = inputs["input"]
        dtype = np().result_type(x.dtype, self.const_dtype)
        x = x.astype(dtype)
        const = self.const_safe.astype(dtype)
        mu = x.data * const

        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            prec = x.precision(raw=True) * self.inv_amp_sq_scalar
            return UA(mu, dtype=dtype, precision=prec, scalar_precision=False)

        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            prec = x.precision(raw=True) * self.inv_amp_sq
            return UA(mu, dtype=dtype, precision=prec, scalar_precision=False)

        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            prec = x.precision(raw=True) * self.inv_amp_sq
            msg_in = UA(mu, dtype=dtype, precision=prec, scalar_precision=False)
            msg_out = self.output_message
            if msg_out is not None:
                qy = (msg_in * msg_out.to_array_precision()).to_scalar_precision()
                return qy / msg_out
            else:
                return msg_in.to_scalar_precision()

        raise RuntimeError("Precision mode not set for forward computation.")

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        dtype = np().result_type(output_msg.dtype, self.const_dtype)
        const = self.const_safe.astype(dtype)
        mu = output_msg.data / const

        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            prec = output_msg.precision(raw=True) / self.inv_amp_sq_scalar
            return UA(mu, dtype=dtype, precision=prec, scalar_precision=True)

        prec = output_msg.precision(raw=True) / self.inv_amp_sq
        msg_array = UA(mu, dtype=dtype, precision=prec, scalar_precision=False)

        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            msg_in = self.input_messages.get(self.inputs["input"]).as_array_precision()
            qx = (msg_array * msg_in).as_scalar_precision()
            return qx / msg_in

        return msg_array

    def forward(self):
        if self.output_message is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            msg = UA.random(
                self.output.event_shape,
                batch_size=self.output.batch_size,
                dtype=self.dtype,
                scalar_precision=(self.output.precision_mode_enum == PrecisionMode.SCALAR),
                rng=self._init_rng,
            )
        else:
            msg = self._compute_forward({"input": self.input_messages[self.inputs["input"]]})
        self.output.receive_message(self, msg)

    def backward(self):
        if self.output_message is None:
            raise RuntimeError("Output message missing.")
        msg = self._compute_backward(self.output_message, exclude="input")
        self.inputs["input"].receive_message(self, msg)

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

        if isinstance(self.const, np().ndarray) and self.const.shape != wave.shape:
            try:
                self.const = np().broadcast_to(self.const, wave.shape)
                self.const_safe = np().broadcast_to(self.const_safe, wave.shape)
                self.inv_amp_sq = np().broadcast_to(self.inv_amp_sq, wave.shape)
            except ValueError:
                raise ValueError(
                    f"MultiplyConstPropagator: constant shape {self.const.shape} not broadcastable to wave shape {wave.shape}"
                )

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(wave.event_shape, batch_size=wave.batch_size, dtype=self.dtype)
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