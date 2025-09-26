from typing import Optional
from ..wave import Wave
from ..factor import Factor
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode
from ...core.backend import np
from .base import Propagator


class ForkPropagator(Propagator):
    """
    Propagator that replicates a single input Wave (batch_size=1)
    into multiple identical outputs with batch_size > 1.
    """

    def __init__(self, batch_size: int, dtype: np().dtype = np().complex64):
        super().__init__(input_names=("input",), dtype=dtype)
        if batch_size < 1:
            raise ValueError("ForkPropagator requires batch_size >= 1.")
        self.batch_size = batch_size

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.batch_size != 1:
            raise ValueError("ForkPropagator only accepts input waves with batch_size=1.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = wave.dtype
        self.event_shape = wave.event_shape

        out_wave = Wave(
            event_shape=self.event_shape,
            batch_size=self.batch_size,
            dtype=self.dtype
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    # -------- Precision mode handling --------
    def _set_precision_mode(self, mode: UnaryPropagatorPrecisionMode):
        if not isinstance(mode, UnaryPropagatorPrecisionMode):
            raise TypeError("ForkPropagator requires UnaryPropagatorPrecisionMode.")
        if mode not in (UnaryPropagatorPrecisionMode.SCALAR, UnaryPropagatorPrecisionMode.ARRAY):
            raise ValueError(f"Unsupported precision mode for ForkPropagator: {mode}")
        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def set_precision_mode_forward(self):
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif x_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    def set_precision_mode_backward(self):
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif y_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    # -------- Message passing (to be implemented) --------
    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Compute forward message. Placeholder implementation.
        Should replicate the input UA using UA.fork().
        """
        raise NotImplementedError("ForkPropagator forward logic not yet implemented.")

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        """
        Compute backward message. Placeholder implementation.
        Should reduce over batch dimension (product_reduce_over_batch).
        """
        raise NotImplementedError("ForkPropagator backward logic not yet implemented.")
