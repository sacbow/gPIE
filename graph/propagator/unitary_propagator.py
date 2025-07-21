import numpy as np
from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.uncertain_array import UncertainArray as UA
from ...core.linalg_utils import reduce_precision_to_scalar
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode


class UnitaryPropagator(Propagator):
    """
    A linear propagator that applies a fixed unitary transformation.

    This propagator models:
        y = U @ x   with U ∈ C^{N * N}, where U is unitary (U.h U = I)

    It supports precision mode conversion between input and output waves:
        - SCALAR → SCALAR
        - SCALAR → ARRAY
        - ARRAY → SCALAR

    Message Passing:
        - Forward: uses output_message and belief to compute message to output
        - Backward: computes x_belief from y and sends message to input

    Belief Fusion:
        In the SCALAR mode, Belief fusion is done using the formula:
            posterior = (tau * U @ x + gamma * r) / (τ + gamma)
        where:
            - r, gamma: input message (mean, precision)
            - p, tau: output message (mean, precision)
        Scalar/array precision is handled accordingly, and can be harmonized.

    Precision Modes:
        - SCALAR: assumes both input and output share scalar precision
        - SCALAR_TO_ARRAY: maps scalar precision input → array output
        - ARRAY_TO_SCALAR: maps array precision input → scalar output

    Args:
        U (np.ndarray): Unitary matrix of shape (N, N).
        precision_mode (UnaryPropagatorPrecisionMode | None): Optional precision mode.
        dtype (np.dtype): Data type (real or complex), default: np.complex128.

    Raises:
        ValueError: If U is not a square 2D unitary matrix.
    """

    def __init__(self, U, precision_mode: Optional[UnaryPropagatorPrecisionMode] = None, dtype=np.complex128):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)

        if U is None:
            raise ValueError("Unitary matrix U must be explicitly provided.")
        self.U = np.asarray(U)
        self.Uh = self.U.conj().T

        if self.U.ndim != 2 or self.U.shape[0] != self.U.shape[1]:
            raise ValueError("U must be a square 2D unitary matrix.")

        self.shape = (self.U.shape[0],)
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None

    def _set_precision_mode(self, mode: str | UnaryPropagatorPrecisionMode):
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)
        allowed = {
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for UnitaryPropagator: {mode}")
        if hasattr(self, "_precision_mode") and self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'"
            )
        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.SCALAR
        elif self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.ARRAY
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        elif self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            return PrecisionMode.SCALAR
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            return PrecisionMode.ARRAY
        return None

    def set_precision_mode_forward(self):
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
        elif x_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    def set_precision_mode_backward(self):
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
        elif y_wave.precision_mode_enum == PrecisionMode.SCALAR:
            if self._precision_mode is None:
                self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    def compute_belief(self):
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)

        if np.issubdtype(msg_x.dtype, np.floating):
            msg_x = msg_x.astype(np.complex128)  # or self.dtype if必要

        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required to compute belief.")

        r = msg_x.data
        p = msg_y.data
        gamma = msg_x._precision
        tau = msg_y._precision

        mode = self._precision_mode
        if mode == UnaryPropagatorPrecisionMode.SCALAR:
            Uh_p = self.Uh @ p
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            self.y_belief = UA(self.U @ x_mean, dtype=self.dtype, precision=denom)

        elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            Ur = self.U @ r
            denom = gamma + tau
            y_mean = (gamma / denom) * Ur + (tau / denom) * p
            self.y_belief = UA(y_mean, dtype=self.dtype, precision=denom)
            scalar_prec = reduce_precision_to_scalar(denom)
            self.x_belief = UA(self.Uh @ y_mean, dtype=self.dtype, precision=scalar_prec)

        elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            Uh_p = self.Uh @ p
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            scalar_prec = reduce_precision_to_scalar(denom)
            self.y_belief = UA(self.U @ x_mean, dtype=self.dtype, precision=scalar_prec)

        else:
            raise ValueError(f"Unknown precision_mode: {self._precision_mode}")

    def forward(self):
        if self.output_message is None or self.y_belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            if self.output.precision_mode == "scalar":
                msg = UA.random(self.shape, dtype=self.dtype, rng=self._init_rng, scalar_precision = True)
            else:
                msg = UA.random(self.shape, dtype=self.dtype, rng=self._init_rng, scalar_precision = False)
        else:
            msg = self.y_belief / self.output_message

        self.output.receive_message(self, msg)

    def backward(self):
        x_wave = self.inputs["input"]
        if self.output_message is None:
            raise RuntimeError("Output message missing.")
        self.compute_belief()
        incoming = self.input_messages[x_wave]
        msg_in = self.x_belief / incoming
        x_wave.receive_message(self, msg_in)

    def set_init_rng(self, rng):
        self._init_rng = rng

    def generate_sample(self, rng):
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        y = self.U @ x
        self.output.set_sample(y)

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.ndim != 1:
            raise ValueError(f"UnitaryPropagator only supports 1D wave input. Got: {wave.shape}")

        self.add_input("input", wave)
        input_gen = wave.generation
        self._set_generation(input_gen + 1)

        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"UProp(gen={gen}, mode={self.precision_mode})"
