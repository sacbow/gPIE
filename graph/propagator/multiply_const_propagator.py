import numpy as np
from typing import Union, Optional
from ..wave import Wave
from .base import Propagator
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode


class MultiplyConstPropagator(Propagator):
    """
    Propagator that multiplies an input wave by a fixed constant (scalar or array).

    This represents a deterministic mapping: y = c * x

    Precision Mode Behavior:
        - If all |const[i]| are approximately equal, input and output precision modes match.
            - SCALAR → SCALAR
            - ARRAY  → ARRAY
        - Otherwise (non-uniform |const|), the output must be array-precision:
            - SCALAR → ARRAY (requires fusion)
            - ARRAY  → ARRAY

    Numerical Stability:
        - Very small-magnitude constants (|c| < 1e-6) are safely regularized to avoid division-by-zero.
        - Precision is scaled by 1 / |c|^2 during belief propagation.

    Attributes:
        uniform (bool): Whether all magnitudes |const| are equal (within tolerance)
        const_safe (ndarray): Const values with small values safely replaced
        inv_amp_sq (ndarray): Precomputed 1 / |const_safe|^2 for precision scaling
    """


    def __init__(self, const: Union[float, complex, np.ndarray]):
        super().__init__(input_names=("input",))
        self.const = np.asarray(const)
        self.const_dtype = self.const.dtype

        # Safe constant to prevent div-by-zero
        abs_vals = np.abs(self.const)
        self.uniform = np.allclose(abs_vals, abs_vals.flat[0], atol=abs_vals.flat[0] * 1e-2)

        # Replace zero or near-zero elements with small magnitude values
        self.const_safe = self.const.copy()
        self.const_safe[abs_vals < 1e-6] = 1e-6 * np.exp(1j * np.angle(self.const_safe[abs_vals < 1e-6]))
        self.inv_amp_sq = 1.0 / np.abs(self.const_safe)**2

    def _set_precision_mode(self, mode: Union[str, UnaryPropagatorPrecisionMode]) -> None:
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)

        if mode not in {
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.ARRAY,
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
        if self.uniform:
            if mode == PrecisionMode.SCALAR:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
            elif mode == PrecisionMode.ARRAY:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        else:
            if mode == PrecisionMode.SCALAR:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
            else:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)

    def set_precision_mode_backward(self) -> None:
        mode = self.output.precision_mode_enum
        if self.uniform:
            if mode == PrecisionMode.ARRAY:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR or self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.SCALAR
        return PrecisionMode.ARRAY

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return PrecisionMode.ARRAY

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Apply multiplicative constant to incoming belief during forward pass.

        - If scalar precision, precision is divided by |const|^2 (scalar).
        - If array precision, each element is divided accordingly.

        Args:
            inputs (dict[str, UA]): Dictionary with one key "input" mapped to UA.

        Returns:
            UncertainArray: Resulting belief with updated mean and scaled precision.
        """

        x = inputs["input"]
        target_dtype = np.result_type(x.dtype, self.const_dtype)
        if x.dtype != target_dtype:
            x = x.astype(target_dtype)
        const = self.const_safe.astype(target_dtype) if self.const_dtype != target_dtype else self.const_safe

        mu = x.data * const
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            precision = x.precision(raw=True) / self.inv_amp_sq[0]
        else:
            precision = x.precision(raw=True) / self.inv_amp_sq

        return UA(mu, dtype=target_dtype, precision=precision)

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        """
        Backward message from output to input wave.

        - Divides output mean by const.
        - Scales precision by |const|^2.
        - If mode is SCALAR_TO_ARRAY, fuses with input message and projects to scalar.

        Args:
            output_msg (UA): Belief from the output wave.
            exclude (str): Name of the input wave to update (always "input" here).

        Returns:
            UA: Backward message for the input wave.
        """

        target_dtype = np.result_type(output_msg.dtype, self.const_dtype)
        const = self.const_safe.astype(target_dtype) if self.const_dtype != target_dtype else self.const_safe

        mu = output_msg.data / const
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            precision = output_msg.precision(raw=True) * self.inv_amp_sq[0]
            return UA(mu, dtype=target_dtype, precision=precision)

        # SCALAR_TO_ARRAY or ARRAY mode
        precision = output_msg.precision(raw=True) * self.inv_amp_sq
        msg_array = UA(mu, dtype=target_dtype, precision=precision)

        target_wave = self.inputs["input"]
        msg_in = self.input_messages.get(target_wave)

        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            # Fuse and downgrade to scalar precision
            q_x = (msg_array * msg_in).as_scalar_precision()
            return q_x / msg_in

        return msg_array


    def __matmul__(self, wave: Wave) -> Wave:
        self.dtype = np.result_type(wave.dtype, self.const_dtype)

        if isinstance(self.const, np.ndarray):
            if self.const.shape != wave.shape:
                try:
                    self.const = np.broadcast_to(self.const, wave.shape)
                    self.const_safe = np.broadcast_to(self.const_safe, wave.shape)
                    self.inv_amp_sq = np.broadcast_to(self.inv_amp_sq, wave.shape)
                except ValueError:
                    raise ValueError(
                        f"MultiplyConstPropagator: constant shape {self.const.shape} not broadcastable to wave shape {wave.shape}"
                    )

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(wave.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def generate_sample(self, rng=None):
        x = self.inputs["input"].get_sample()
        if x is not None:
            const = self.const.astype(x.dtype) if self.const_dtype != x.dtype else self.const
            self.output.set_sample(x * const)
