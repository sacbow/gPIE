import numpy as np
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_unitary_matrix, reduce_precision_to_scalar


class UnitaryPropagator(Propagator):
    def __init__(self, U=None, shape=None, dtype=np.complex128, seed=None):
        super().__init__(input_names=("input",), dtype=dtype)

        if U is not None:
            self.U = U
        elif shape is not None:
            n = shape[0] if isinstance(shape, tuple) else shape
            self.U = random_unitary_matrix(n, seed=seed, dtype=dtype)
        else:
            raise ValueError("Either U or shape must be provided.")

        if self.U.ndim != 2 or self.U.shape[0] != self.U.shape[1]:
            raise ValueError("U must be a square 2D unitary matrix.")

        self.shape = (self.U.shape[0],)
        self.add_input("input", Wave(self.shape, dtype=dtype))

        self.x_belief = None
        self.y_belief = None

    def compute_belief(self):
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required to compute belief.")

        r = msg_x.data
        gamma = msg_x.to_scalar_precision()

        p = msg_y.data
        tau = msg_y.precision

        Ur = self.U @ r
        denom = gamma + tau
        y_mean = (gamma / denom) * Ur + (tau / denom) * p

        scalar_prec = reduce_precision_to_scalar(denom)
        self.y_belief = UA(y_mean, dtype=self.dtype, precision=scalar_prec)

        x_mean = self.U.conj().T @ y_mean
        self.x_belief = UA(x_mean, dtype=self.dtype, precision=scalar_prec)

    def forward(self):
        if self.y_belief is None or self.output_message is None:
            msg = UA.random(self.shape, dtype=self.dtype)
        else:
            msg = self.y_belief / self.output_message
        self.output.receive_message(self, msg)

    def backward(self):
        x_wave = self.inputs["input"]
        if self.output_message is None:
            raise RuntimeError("Output message not available for backward propagation.")

        self.compute_belief()
        msg_in = self.x_belief / UA.as_scalar_precision(self.input_messages[x_wave])
        x_wave.receive_message(self, msg_in)

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.ndim != 1:
            raise ValueError(f"UnitaryPropagator only supports 1D wave input. Got: {wave.shape}")

        self.add_input("input", wave)

        input_gen = wave._generation 
        self._set_generation(input_gen + 1)

        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave

        return self.output
    
    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        raise NotImplementedError("UnitaryPropagator uses custom forward().")

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        raise NotImplementedError("UnitaryPropagator uses custom backward().")

