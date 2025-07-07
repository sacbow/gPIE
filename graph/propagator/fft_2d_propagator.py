import numpy as np
from .base import Propagator
from core.uncertain_array import UncertainArray as UA
from core.linalg_utils import reduce_precision_to_scalar
from graph.wave import Wave

class FFT2DPropagator(Propagator):
    def __init__(self, shape, scalar_precision="output", dtype=np.complex128):
        super().__init__(input_names=("input",), dtype=dtype)
        self.shape = shape
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None
        self.scalar_precision = scalar_precision

    def set_init_rng(self, rng):
        self._init_rng = rng

    def _fft2_centered(self, x):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

    def _ifft2_centered(self, y):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y), norm="ortho"))

    def compute_belief(self):
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required.")

        r = msg_x.data
        p = msg_y.data
        gamma = msg_x.precision
        tau = msg_y.precision

        if self.scalar_precision == "input":
            Ur = self._fft2_centered(r)
            denom = gamma + tau
            y_mean = (gamma / denom) * Ur + (tau / denom) * p
            self.y_belief = UA(y_mean, dtype=self.dtype, precision=denom)
            scalar_prec = reduce_precision_to_scalar(denom)
            self.x_belief = UA(self._ifft2_centered(y_mean), dtype=self.dtype, precision=scalar_prec)

        elif self.scalar_precision == "output":
            Uh_p = self._ifft2_centered(p)
            denom = gamma + tau
            x_mean = (gamma / denom) * r + (tau / denom) * Uh_p
            self.x_belief = UA(x_mean, dtype=self.dtype, precision=denom)
            scalar_prec = reduce_precision_to_scalar(denom)
            self.y_belief = UA(self._fft2_centered(x_mean), dtype=self.dtype, precision=scalar_prec)

        else:
            raise ValueError("Unknown scalar_precision mode")

    def forward(self):
        if self.output_message is None or self.y_belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            msg = UA.random(self.shape, dtype=self.dtype, rng=self._init_rng)
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

    def __matmul__(self, wave: Wave) -> Wave:
        if wave.ndim != 2:
            raise ValueError("FFT2DPropagator only supports 2D wave input.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def generate_sample(self, rng):
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        y = self._fft2_centered(x)
        self.output.set_sample(y)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"FFT2DProp(gen={gen})"
