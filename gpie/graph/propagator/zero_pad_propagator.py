from typing import Optional
from ..wave import Wave
from .base import Propagator
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode
from ...core.backend import np

class ZeroPadPropagator(Propagator):
    """
    Apply zero-padding to an input wave along event dimensions.

    Args:
        pad_width (tuple[tuple[int,int], ...]):
            Padding spec in NumPy style, but excluding batch dimension.
            Example: ((2,2), (3,3)) → pad 2 rows top/bottom and 3 cols left/right.
    """

    def __init__(self, pad_width: tuple[tuple[int, int], ...]):
        super().__init__(input_names=("input",), precision_mode=UnaryPropagatorPrecisionMode.ARRAY)
        self.pad_width = pad_width

    def __matmul__(self, wave: Wave) -> Wave:
        if len(self.pad_width) != len(wave.event_shape):
            raise ValueError(
                f"pad_width length {len(self.pad_width)} does not match event_shape rank {len(wave.event_shape)}"
            )

        # compute new shape
        new_shape = tuple(
            dim + left + right for dim, (left, right) in zip(wave.event_shape, self.pad_width)
        )

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        out_wave = Wave(
            event_shape=new_shape,
            batch_size=wave.batch_size,
            dtype=wave.dtype,
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return out_wave

    def get_input_precision_mode(self, wave: Wave) -> PrecisionMode:
        return PrecisionMode.ARRAY

    def get_output_precision_mode(self) -> PrecisionMode:
        return PrecisionMode.ARRAY

    def set_precision_mode_forward(self):
        return

    def set_precision_mode_backward(self):
        return

    def _compute_forward(self, inputs: dict[str, UA], block = None) -> UA:
        x_msg = inputs["input"]
        return x_msg.zero_pad(self.pad_width)

    def _compute_backward(self, output_msg: UA, exclude: str, block = None) -> UA:
        """
        Crop the padded UncertainArray back to the input shape.
        """
        input_shape = self.inputs["input"].event_shape

        # build slicing indices according to pad_width
        idx = tuple(slice(l, l + dim) for (l, _), dim in zip(self.pad_width, input_shape))

        cropped_data = output_msg.data[(slice(None),) + idx]
        cropped_prec = output_msg.precision(raw=False)[(slice(None),) + idx]

        return UA(cropped_data, dtype=output_msg.dtype, precision=cropped_prec)

    
    def get_sample_for_output(self, rng=None):
        """
        Return zero-padded sample from the input wave.

        The output has the same shape as the output wave:
            (batch_size, *padded_event_shape)

        Raises:
            RuntimeError: if the input wave has no sample set.
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")

        # Pad only event dimensions (exclude batch dim)
        pad_full = ((0, 0),) + self.pad_width
        return np().pad(x, pad_full, mode="constant", constant_values=0)

