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

    Internal state:
        - self.belief: UA shaped like the *input* (batch_size=1, event_shape)
        - self.m_in_old: last input UA used to update `belief` incrementally

    Note on caching the previous input:
        We cache the previous input message by *reference* (shallow copy).
        `Factor.receive_message()` replaces the stored UA object on each update
        (no in-place mutation), so keeping the old UA reference is safe and avoids
        unnecessary deep copies of large arrays.
    """

    def __init__(self, batch_size: int, dtype: np().dtype = np().complex64):
        super().__init__(input_names=("input",), dtype=dtype)
        if batch_size < 1:
            raise ValueError("ForkPropagator requires batch_size >= 1.")
        self.batch_size = batch_size

        # Internal EP state
        self.belief: Optional[UA] = None
        self.m_in_old: Optional[UA] = None

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
    def _compute_forward(self, inputs: Dict[str, UA]) -> UA:
        """
        Compute forward message to the output.

        Logic:
            - If first iteration: initialize belief with current input and fork it.
            - Else: update belief incrementally using cached previous input (by reference):
                    belief_new = (belief_old / m_in_old) * m_in_new
              Then send message to output:
                    msg_out = belief_new.fork(batch_size) / m_in_new

        Returns:
            UA: message to the output wave (batch_size=self.batch_size).
        """
        m_in = inputs["input"]

        # First iteration (no belief yet): initialize and fork to output
        if self.belief is None or self.output_message is None:
            self.belief = m_in
            self.m_in_old = m_in  # cache by reference; old UA remains valid until GC
            return m_in.fork(batch_size=self.batch_size)

        # Incremental update of belief using cached previous input
        # belief_new = (belief_old / m_in_old) * m_in
        self.belief = (self.belief / self.m_in_old) * m_in  # type: ignore[arg-type]
        # Update cache for next iteration (by reference; no deep copy)
        self.m_in_old = m_in
        # Outgoing message to output (replicate and divide out current input)
        return self.belief.fork(batch_size=self.batch_size) / self.output_message

    def _compute_backward(self, output_msg: UA, exclude: str) -> UA:
        """
        Compute backward message to the (only) input.

        Logic:
            - Recompute belief without incremental shortcut:
                  belief = m_in * product_reduce_over_batch(m_out)
            - Send message to input by cancelling out the output message:
                  msg_in = belief.fork(batch_size) / m_out

        Args:
            output_msg: UA from the output side (batch_size=self.batch_size).
            exclude:   Unused (required by signature); only one input exists.

        Returns:
            UA: message to the input wave (batch_size=self.batch_size).
        """
        # Retrieve current input message stored on the factor
        m_in = self.input_messages[self.inputs["input"]]
        m_out = output_msg

        # Full recomputation of belief (no incremental trick needed here)
        self.belief = m_in * m_out.product_reduce_over_batch()

        # Backward message to input has the same batch_size as m_out
        return self.belief / m_in
