from ...core.backend import np
from typing import Optional
from ..wave import Wave
from .base import Propagator
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode
from ...core.accumulative_uncertain_array import AccumulativeUncertainArray as AUA
from ...core.uncertain_array import UncertainArray as UA

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
            self.indices = [indices]  # single slice → list
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

        # AUA will be initialized later, once we know input dtype/event_shape
        self.output_product: Optional[AUA] = None
    
    def to_backend(self) -> None:
        """
        Ensures that this propagator and its associated AccumulativeUncertainArray (AUA)
        remain consistent when switching between NumPy and CuPy backends.
        """
        # move associated AUA (if initialized)
        if self.output_product is not None:
            self.output_product.to_backend()


    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to an input wave and construct the output wave.

        This performs the graph-building step: the propagator registers its input,
        validates that all slice indices are compatible with the input's event_shape,
        and then creates the output wave.

        Args:
            wave (Wave): Input wave. Must have batch_size=1.

        Returns:
            Wave: Output wave with event_shape = self.patch_shape and
                batch_size = len(self.indices).

        Raises:
            ValueError: if batch_size != 1, slice rank mismatch, or indices
                        are out of range for the input's event_shape.
        """

        # input wave must have batch_size == 1
        if wave.batch_size != 1:
            raise ValueError("SlicePropagator only accepts input waves with batch_size=1.")

        # check that indices fit within wave.event_shape
        for idx in self.indices:
            if len(idx) != len(wave.event_shape):
                raise ValueError(
                    f"Slice rank mismatch: got {len(idx)} slices, "
                    f"but wave.event_shape={wave.event_shape}"
                )
            for s, dim in zip(idx, wave.event_shape):
                if s.start < 0 or s.stop > dim:
                    raise ValueError(
                        f"Slice {s} out of range for dimension {dim} "
                        f"(wave.event_shape={wave.event_shape})"
                    )

        # register input
        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)

        # output wave: batch_size = number of slices, event_shape = patch_shape
        self.dtype = wave.dtype
        out_wave = Wave(
            event_shape=self.patch_shape,
            batch_size=len(self.indices),
            dtype=self.dtype
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        self.output_product = AUA(
            event_shape=wave.event_shape,
            indices=self.indices,
            dtype=self.dtype
        )
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
    
    def _compute_forward(self, inputs: dict[str, UA]) -> UA:
        """
        Compute forward message: input → patches.

        The input message (UA, batch_size=1) is fused into an internal
        AccumulativeUncertainArray (AUA). Each slice defined in self.indices
        is then extracted as a patch, and all patches are stacked into a
        batched UncertainArray.

        Args:
            inputs (dict[str, UA]): {"input": UA}, batch_size must be 1.

        Returns:
            UA: Batched UncertainArray with shape (num_patches, *patch_shape).

        Raises:
            ValueError: if the input UA does not have batch_size=1.
        """

        x_msg = inputs["input"]
        if x_msg.batch_size != 1:
            raise ValueError("SlicePropagator expects batch_size=1 input message.")

        if self.output_message is None:
            msg_to_send = x_msg.extract_patches(self.indices)
            return msg_to_send

        # Fuse input UA into AUA
        self.output_product.mul_ua(x_msg)
        msg_to_send = self.output_product.extract_patches() / self.output_message
        # Return extracted patches
        return msg_to_send

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        """
        Compute backward message: patches → input.

        The output message (batched UA over patches) is scattered back into
        an AccumulativeUncertainArray, reconstructing a full-size UA that
        matches the input wave.

        Args:
            output_msg (UA): Batched UncertainArray with batch_size == len(indices).
            exclude (str): Ignored for SlicePropagator (single input only).

        Returns:
            UA: Reconstructed input UncertainArray (batch_size=1).

        Raises:
            ValueError: if output_msg batch_size != len(indices).
            RuntimeError: if forward has not been run before backward.
        """

        if output_msg.batch_size != len(self.indices):
            raise ValueError("Output message batch size mismatch.")

        if self.output_product is None:
            raise RuntimeError("Forward pass must be run before backward.")

        # Reset AUA
        self.output_product.clear()

        # Scatter patches back into full-size AUA
        self.output_product.scatter_mul(output_msg)

        # Return reconstructed UA for input
        return self.output_product.as_uncertain_array()