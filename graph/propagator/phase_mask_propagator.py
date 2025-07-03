import numpy as np
from .base import Propagator
from core.uncertain_array import UncertainArray as UA
from graph.wave import Wave

class PhaseMaskPropagator(Propagator):
    """
    Component-wise complex phase modulation propagator.
    Applies a fixed unit-magnitude phase mask to the wave.
    """

    def __init__(self, phase_mask, dtype=np.complex128):
        """
        Args:
            phase_mask (ndarray): Complex array of shape (H, W) with unit magnitude.
            dtype (np.dtype): Complex dtype (default: complex128).
        """
        super().__init__(input_names=("input",), dtype=dtype)

        if not np.allclose(np.abs(phase_mask), 1.0):
            raise ValueError("Phase mask must have unit magnitude.")

        self.phase_mask = phase_mask
        self.shape = phase_mask.shape

    def _compute_forward(self, incoming: dict[str, UA]) -> UA:
        """
        Forward message: multiply input by phase mask (component-wise).
        """
        ua = incoming["input"]
        return UA(ua.data * self.phase_mask, dtype=ua.dtype, precision=ua._precision)

    def _compute_backward(self, outgoing: UA, exclude: str = None) -> UA:
        """
        Backward message: divide by phase mask (component-wise inverse).
        """
        return UA(outgoing.data / self.phase_mask, dtype=outgoing.dtype, precision=outgoing._precision)

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to a Wave using @ operator.
        """
        if wave.shape != self.shape:
            raise ValueError("Input wave shape does not match phase mask shape.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(self.shape, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def generate_sample(self, rng):
        """
        Generate output sample by applying the phase mask to input sample.
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        y = x * self.phase_mask
        self.output.set_sample(y)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"PhaseMaskProp(gen={gen})"
