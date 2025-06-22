import numpy as np
from .base import Propagator
from graph.wave import Wave
from core.uncertain_array import UncertainArray as UA

class MatrixPropagator(Propagator):
    def __init__(self, mat: np.ndarray, dtype=np.complex128):
        """
        mat: Square matrix representing linear transformation (NxN)
        """
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("MatrixPropagator only supports square matrices.")
        self.mat = mat
        self.shape = mat.shape  # (N, N)
        super().__init__(input_names=("input",), dtype=dtype)

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to an input wave using the @ operator.
        Returns the output wave that this propagator produces.
        """
        self.add_input("input", wave)  # Connect input
        out = Wave(wave.shape, dtype=self.dtype)  # Create output Wave
        self.set_output(out)  # Connect output (and set generation)
        return out

    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        raise NotImplementedError("Forward logic not implemented yet.")

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        raise NotImplementedError("Backward logic not implemented yet.")
