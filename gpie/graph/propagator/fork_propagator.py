from typing import Optional, Dict
from ..wave import Wave
from ..factor import Factor
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode
from ...core.backend import np
from .base import Propagator


class ForkPropagator(Propagator):
    """
    Replicates a single input Wave (batch_size=1) into a batched output
    with `batch_size > 1`.

    Precision-mode policy:
        - Supported: SCALAR, ARRAY
        - Not supported: SCALAR_TO_ARRAY, ARRAY_TO_SCALAR


    Note on caching the previous input:
        We cache the reduced product of the message from output previous input message by self.child_product.
    """

    def __init__(self, batch_size: int, dtype: np().dtype = np().complex64):
        super().__init__(input_names=("input",), dtype=dtype)
        if batch_size < 1:
            raise ValueError("ForkPropagator requires batch_size >= 1.")
        self.batch_size = batch_size

        # cache
        self.child_product: Optional[UA] = None

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect the input wave and create an output wave with replicated batch size.
        """
        if wave.batch_size != 1:
            raise ValueError("ForkPropagator only accepts input waves with batch_size=1.")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = wave.dtype
        self.event_shape = wave.event_shape

        out_wave = Wave(
            event_shape=self.event_shape,
            batch_size=self.batch_size,
            dtype=self.dtype
        )
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    # -------- Precision mode handling --------
    def _set_precision_mode(self, mode: UnaryPropagatorPrecisionMode):
        """
        Restrict precision mode to SCALAR or ARRAY for fork.
        """
        if not isinstance(mode, UnaryPropagatorPrecisionMode):
            raise TypeError("ForkPropagator requires UnaryPropagatorPrecisionMode.")
        if mode not in (UnaryPropagatorPrecisionMode.SCALAR, UnaryPropagatorPrecisionMode.ARRAY):
            raise ValueError(f"Unsupported precision mode for ForkPropagator: {mode}")
        # Delegate to base consistency checks if needed
        super()._set_precision_mode(mode)

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY:
            return PrecisionMode.ARRAY
        elif self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR:
            return PrecisionMode.SCALAR
        return None

    def set_precision_mode_forward(self):
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif x_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    def set_precision_mode_backward(self):
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
        elif y_wave.precision_mode_enum == PrecisionMode.SCALAR:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    # -------- Message passing --------
    def _compute_forward(self, inputs: Dict[str, UA], block = None) -> UA:
        """
        Compute forward message to the output.
        Returns:
            UA: message to the output wave (batch_size=self.batch_size).
        """
        m_in = inputs["input"]

        # First iteration (no belief yet): initialize and fork to output
        if self.child_product is None or self.output_message is None:
            msg = m_in.fork(batch_size=self.batch_size)
            return msg
        # compute belief
        belief = m_in * self.child_product
        msg = belief.fork(batch_size=self.batch_size) / self.output_message
        return msg

    def _compute_backward(self, output_msg: UA, exclude: str, block = None) -> UA:
        """
        Compute backward message to the (only) input.

        Args:
            output_msg: UA from the output side (batch_size=self.batch_size).
            exclude:   Unused (required by signature); only one input exists.

        Returns:
            UA: message to the input wave (batch_size=self.batch_size).
        """
        m_out = output_msg.product_reduce_over_batch()
        self.child_product = m_out
        return m_out
    
    def get_sample_for_output(self, rng):
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        forked_sample = np().broadcast_to(x, (self.batch_size,) + x_wave.event_shape).copy()
        return forked_sample