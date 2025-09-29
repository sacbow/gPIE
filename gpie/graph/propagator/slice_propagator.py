from ...core.backend import np
from ..wave import Wave
from .base import Propagator
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode

class SlicePropagator(Propagator):
    """
    Extracts fixed-size patches from a single input wave.

    Each patch is defined by a tuple of `slice` objects. Multiple patches
    can be specified by passing a list of slice-tuples, in which case the
    output wave has one batch entry per patch.

    Constraints:
        - Input wave must have `batch_size == 1`
        - All provided slices must yield the same patch shape

    Example:
        >>> x = Wave(event_shape=(32, 32), batch_size=1)
        >>> prop = SlicePropagator([(slice(0, 16), slice(0, 16)),
        ...                         (slice(16, 32), slice(16, 32))])
        >>> y = prop @ x
        >>> y.batch_size   # 2 patches
        2
        >>> y.event_shape  # (16, 16)
        (16, 16)
    """

    def __init__(self, indices):
        super().__init__(input_names=("input",), precision_mode = UnaryPropagatorPrecisionMode.ARRAY)
        # normalize indices to a list of tuples of slices
        if isinstance(indices, tuple) and all(isinstance(s, slice) for s in indices):
            self.indices = [indices]  # single slice → list化
        elif isinstance(indices, list):
            if not all(isinstance(idx, tuple) and all(isinstance(s, slice) for s in idx) for idx in indices):
                raise TypeError("indices must be a tuple of slices or a list of tuples of slices.")
            self.indices = indices
        else:
            raise TypeError("indices must be a tuple of slices or a list of tuples of slices.")

        # check shapes
        shapes = [tuple(s.stop - s.start for s in idx) for idx in self.indices]
        if not all(sh == shapes[0] for sh in shapes):
            raise ValueError("All slice indices must produce patches of the same shape.")
        self.patch_shape = shapes[0]


    def __matmul__(self, wave: Wave) -> Wave:
        # input wave must have batch_size == 1
        if wave.batch_size != 1:
            raise ValueError("SlicePropagator only accepts input waves with batch_size=1.")

        # register input
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        # ② output wave: batch_size = number of slices, event_shape = patch_shape
        self.dtype = wave.dtype
        out_wave = Wave(
            event_shape=self.patch_shape,
            batch_size=len(self.indices),
            dtype=self.dtype
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output
    

    def get_sample_for_output(self, rng=None):
        """
        Return deterministic patches from the input sample.

        Uses the provided slice indices to extract patches from the input
        wave's sample. The resulting array has shape:
            (num_patches, *patch_shape)

        Raises:
            RuntimeError: if the input wave has no sample set.
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")

        patches = [x[(0,) + idx] for idx in self.indices]
        return np().stack(patches, axis=0)
    
    
    def set_precision_mode_forward(self):
        return

    def set_precision_mode_backward(self):
        return

    def get_input_precision_mode(self, wave: Wave) -> PrecisionMode:
        return PrecisionMode.ARRAY

    def get_output_precision_mode(self) -> PrecisionMode:
        return PrecisionMode.ARRAY
