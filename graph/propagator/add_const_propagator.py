from ..wave import Wave
from .base import Propagator
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode
import numpy as np
from typing import Union, Optional


class AddConstPropagator(Propagator):
    """
    Propagator that adds a fixed constant (scalar or array) to the input Wave.

    This operator models a deterministic addition:
        y = x + c

    Supports:
        - Scalar (float, complex) or array-valued constants
        - Broadcasting of const to match Wave shape
        - Safe dtype promotion using np.result_type
        - Preserves input precision mode (scalar/array)
        - Compatible with EP-style forward/backward message passing

    Example:
        >>> x = ~SupportPrior(...)
        >>> y = x + 3.0
    """

    def __init__(self, const: Union[float, complex, np.ndarray]):
        super().__init__(input_names=("input",))
        self.const = np.asarray(const)
        self.const_dtype = self.const.dtype

    def _set_precision_mode(self, mode: Union[str, PrecisionMode]) -> None:
        """
        Assign precision mode (must be scalar or array). Raises on conflict.
        """
        if isinstance(mode, str):
            mode = PrecisionMode(mode)

        if mode.value not in ("scalar", "array"):
            raise ValueError(f"Invalid precision mode for AddConstPropagator: {mode}")

        if self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict for AddConstPropagator: "
                f"existing='{self._precision_mode}', requested='{mode}'"
            )

        self._precision_mode = mode

    def set_precision_mode_forward(self) -> None:
        """
        Propagate precision mode from input to output.
        """
        mode = self.inputs["input"].precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode)
            self.output._set_precision_mode(mode)

    def set_precision_mode_backward(self) -> None:
        """
        Propagate precision mode from output to input.
        """
        mode = self.output.precision_mode_enum
        if mode is not None:
            self._set_precision_mode(mode)
            self.inputs["input"]._set_precision_mode(mode)

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        return self._precision_mode

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        return self._precision_mode

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Add constant to input data with correct dtype promotion and precision adjustment.
        """
        x = inputs["input"]
        target_dtype = np.result_type(x.dtype, self.const_dtype)

        # Promote UA to target_dtype, with proper precision scaling
        if x.dtype != target_dtype:
            x = x.astype(target_dtype)

        # Promote const to target_dtype
        const = self.const.astype(target_dtype) if self.const_dtype != target_dtype else self.const

        return UA(x.data + const, dtype=target_dtype, precision=x.precision(raw=True))


    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        """
        Subtract constant from output message (inverse of forward).
        """
        const = self.const.astype(output_msg.dtype) if output_msg.dtype != self.const_dtype else self.const
        return UA(output_msg.data - const, dtype=output_msg.dtype, precision=output_msg.precision(raw=True))

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect the propagator to a Wave and create an output Wave.
        Automatically promotes dtype and handles shape broadcasting.
        """
        self.dtype = np.result_type(wave.dtype, self.const_dtype)

        if isinstance(self.const, np.ndarray):
            if self.const.shape != wave.shape:
                try:
                    self.const = np.broadcast_to(self.const, wave.shape)
                except ValueError:
                    raise ValueError(
                        f"AddConstPropagator: constant shape {self.const.shape} not broadcastable to wave shape {wave.shape}"
                    )

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(wave.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def generate_sample(self, rng=None):
        """
        Generate a sample by adding constant to input sample.
        """
        x_sample = self.inputs["input"].get_sample()
        if x_sample is not None:
            const = self.const.astype(x_sample.dtype) if x_sample.dtype != self.const_dtype else self.const
            self.output.set_sample(x_sample + const)
