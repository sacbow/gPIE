import numpy as np
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import random_unitary_matrix


class UnitaryPropagator(Propagator):
    def __init__(self, U=None, shape=None, dtype=np.complex128, seed=None):
        """
        Unitary linear propagator y = Ux.

        Args:
            U (np.ndarray): Unitary matrix of shape (n, n). If None, random unitary will be generated.
            shape (tuple): Shape of input/output wave (used if U is None).
            dtype (np.dtype): Complex data type.
            seed (int): Optional random seed for generating U.
        """
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

        self.shape = (self.U.shape[0],)  # Ensure 1D wave shape

        # Create and connect input/output waves
        self.add_input("input", Wave(self.shape, dtype=dtype))
        self.connect_output(Wave(self.shape, dtype=dtype))

    def _compute_forward(self, messages):
        ua_x = messages["input"]
        r = ua_x.data
        gamma = ua_x.to_scalar_precision()

        y_mean = self.U @ r
        return UA(y_mean, precision=gamma)

    def _compute_backward(self, messages):
        ua_y = messages["output"]
        p = ua_y.data
        tau = ua_y.to_scalar_precision()

        x_mean = self.U.conj().T @ p
        return UA(x_mean, precision=tau)

    def forward(self):
        input_wave = self.inputs["input"]
        msg_in = self.input_messages.get(input_wave)

        if msg_in is None:
            raise RuntimeError("Input message not available for forward propagation.")

        messages = {"input": msg_in}
        msg_out = self._compute_forward(messages)
        self.output.receive_message(self, msg_out)

    def backward(self):
        msg_out = self.output_message
        if msg_out is None:
            raise RuntimeError("Output message not available for backward propagation.")

        messages = {"output": msg_out}
        msg_in = self._compute_backward(messages)

        input_wave = self.inputs["input"]
        self.input_messages[input_wave] = msg_in
        input_wave.receive_message(self, msg_in)

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Enables syntax: Y = UnitaryPropagator(U) @ X

        This registers the input wave and returns the output wave.
        """
        if wave.ndim != 1:
            raise ValueError(f"UnitaryPropagator only supports 1D wave input. Got: {wave.shape}")

        self.add_input("input", wave)

        input_gen = wave.generation if wave.generation is not None else 0
        self.set_generation(input_gen + 1)

        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave.set_generation(self.generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave

        return self.output