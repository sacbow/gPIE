from typing import Optional
from .base import Propagator
from ..wave import Wave
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.fft import get_fft_backend
from ...core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_complex_dtype


class FFT2DPropagator(Propagator):
    """
    A centered 2D FFT-based propagator for EP message passing.

    It defines a unitary mapping between spatial and frequency domain using:
        y = FFT2_centered(x)
        x = IFFT2_centered(y)

    Supports:
        - SCALAR <-> SCALAR
        - SCALAR <-> ARRAY
        - ARRAY → SCALAR

    Precision handling follows `UnaryPropagatorPrecisionMode`.

    Notes:
        - Assumes event_shape is 2D (e.g., (H, W))
        - Internally uses fftshifted FFTs
    """

    def __init__(
        self,
        event_shape: Optional[tuple[int, int]] = None,
        precision_mode: Optional[UnaryPropagatorPrecisionMode] = None,
        dtype=np().complex64,
    ):
        super().__init__(input_names=("input",), dtype=dtype, precision_mode=precision_mode)
        self.event_shape = event_shape
        self._init_rng = None
        self.x_belief: Optional[UA] = None
        self.y_belief: Optional[UA] = None

    def to_backend(self):
        """Synchronize dtype with current backend."""
        self.dtype = np().dtype(self.dtype)

    def _set_precision_mode(self, mode: str | UnaryPropagatorPrecisionMode):
        """Restrict precision modes to the ones supported by FFT2DPropagator."""
        if isinstance(mode, str):
            mode = UnaryPropagatorPrecisionMode(mode)
        allowed = {
            UnaryPropagatorPrecisionMode.SCALAR,
            UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
            UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
        }
        if mode not in allowed:
            raise ValueError(f"Invalid precision_mode for FFT2DPropagator: {mode}")
        if hasattr(self, "_precision_mode") and self._precision_mode is not None and self._precision_mode != mode:
            raise ValueError(
                f"Precision mode conflict: existing='{self._precision_mode}', new='{mode}'"
            )
        self._precision_mode = mode

    def get_input_precision_mode(self, wave: Wave) -> Optional[PrecisionMode]:
        """
        Return the expected precision mode of the input wave, given self._precision_mode.
        """
        if self._precision_mode is not None:
            if self._precision_mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
                return PrecisionMode.ARRAY
            else:
                return PrecisionMode.SCALAR
        return None

    def get_output_precision_mode(self) -> Optional[PrecisionMode]:
        """
        Return the expected precision mode of the output wave, given self._precision_mode.
        """
        if self._precision_mode is not None:
            if self._precision_mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
                return PrecisionMode.ARRAY
            else:
                return PrecisionMode.SCALAR
        return None

    def set_precision_mode_forward(self):
        """
        Infer propagator precision mode from the input wave (forward pass).
        """
        x_wave = self.inputs["input"]
        if x_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)

    def set_precision_mode_backward(self):
        """
        Infer propagator precision mode from the output wave (backward pass).
        """
        y_wave = self.output
        if y_wave.precision_mode_enum == PrecisionMode.ARRAY:
            self._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    def compute_belief(self, block=None):
        """
        Compute the joint belief over (x, y) under the unitary transform y = F x.

        This method updates the internal variables:
            - self.x_belief
            - self.y_belief

        Block semantics:
            - If block is None:
                  Compute the full-batch belief (legacy behavior).
            - If block is a slice:
                  Only compute the belief for that batch slice,
                  and merge the results into the existing full-batch beliefs
                  via UA.insert_block().
        """
        x_wave = self.inputs["input"]
        msg_x = self.input_messages[x_wave]
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required to compute belief.")

        # Extract block (full batch if block=None)
        msg_x_block = msg_x.extract_block(block)
        msg_y_block = msg_y.extract_block(block)

        # Ensure complex dtype compatibility
        if not np().issubdtype(msg_x_block.data.dtype, np().complexfloating):
            msg_x_block = msg_x_block.astype(self.dtype)

        mode = self._precision_mode

        # Compute belief on block
        if mode == UnaryPropagatorPrecisionMode.SCALAR:
            x_block = msg_x_block * msg_y_block.ifft2_centered()
            y_block = x_block.fft2_centered()

        elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            y_block = msg_x_block.fft2_centered().as_array_precision() * msg_y_block
            x_block = y_block.ifft2_centered()

        elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            x_block = msg_x_block * msg_y_block.ifft2_centered().as_array_precision()
            y_block = x_block.fft2_centered()

        else:
            raise ValueError(f"Unknown precision_mode: {self._precision_mode}")

        # ------------------------------------------------------------
        # Lazy init of full-batch beliefs: precision mode from propagator
        # ------------------------------------------------------------
        if self.x_belief is None or self.y_belief is None:
            if mode == UnaryPropagatorPrecisionMode.SCALAR:
                scalar_x = True
                scalar_y = True
            elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
                scalar_x = True
                scalar_y = False
            elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
                scalar_x = False
                scalar_y = True
            else:
                raise ValueError(f"Unknown precision_mode: {mode}")

            if self.x_belief is None:
                self.x_belief = UA.zeros(
                    event_shape=self.event_shape,
                    batch_size=x_wave.batch_size,
                    dtype=self.dtype,
                    precision=1.0,
                    scalar_precision=scalar_x,
                )
            if self.y_belief is None:
                self.y_belief = UA.zeros(
                    event_shape=self.event_shape,
                    batch_size=self.output.batch_size,
                    dtype=self.dtype,
                    precision=1.0,
                    scalar_precision=scalar_y,
                )

        # Merge block into full beliefs
        self.x_belief.insert_block(block, x_block)
        self.y_belief.insert_block(block, y_block)

        # Return block belief (needed for backward)
        return x_block, y_block

    def _compute_forward(self, inputs, block=None):
        """
        Compute the outgoing EP message in the forward direction (x → y).

        This method implements:
            - Initial forward pass:
                  m_y = FFT2_centered(m_x)
            - Steady-state EP update:
                  m_y = y_belief / m_y_old

        Block semantics:
            - If block is None:
                  Operate over full batch (legacy).
            - If block is a slice:
                  Extract only the block of the inputs and apply the
                  same EP update rule to that block.

        Returns:
            msg_block (UA):
                The outgoing message restricted to the given block,
                which will be merged by Propagator.forward().
        """
        msg_x = inputs["input"]
        out_msg = self.output_message
        yb = self.y_belief

        # Extract block (no-op if block=None)
        msg_x_block = msg_x.extract_block(block)

        # Case A: initial forward message
        if out_msg is None and yb is None:
            msg = msg_x_block.fft2_centered()
            if self.output.precision_mode_enum == PrecisionMode.ARRAY:
                msg = msg.as_array_precision()
            return msg

        # Case B: steady-state EP update
        out_msg_block = out_msg.extract_block(block)
        yb_block = yb.extract_block(block)
        return yb_block / out_msg_block


    def _compute_backward(self, output_msg, exclude, block=None):
        """
        Compute the EP backward message (y → x).

        Implements:
            m_x = x_belief / m_x_old

        Block semantics:
            - compute_belief(block) updates only the relevant batch slice,
              and returns (x_belief_block, y_belief_block).
            - We use only x_belief_block here.
            - m_x_old is sliced consistently by extract_block(block).

        Args:
            output_msg (UA):
                The current outgoing message from y.
            exclude (str):
                Must be "input" for FFT2DPropagator.
            block (slice or None):
                Block of the batch dimension to update.

        Returns:
            msg_block (UA):
                Backward message restricted to the given block.
        """
        if exclude != "input":
            raise RuntimeError("FFT2DPropagator has only one input: 'input'.")

        # Compute only the relevant block of beliefs
        x_block, _ = self.compute_belief(block=block)

        wave = self.inputs["input"]
        msg_x_old = self.input_messages[wave]
        msg_x_old_block = msg_x_old.extract_block(block)

        return x_block / msg_x_old_block



    def set_init_rng(self, rng):
        """Set RNG for possible future initialization needs."""
        self._init_rng = rng

    def get_sample_for_output(self, rng):
        """
        Generate a sample for the output wave given a sample from the input wave.

        This simply applies the centered 2D FFT to the input sample.
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        fft = get_fft_backend()
        return fft.fft2_centered(x)

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Connect this propagator to an input Wave and create an output Wave.

        The output has the same event_shape and batch_size, with complex dtype
        promoted as needed.
        """
        if len(wave.event_shape) != 2:
            raise ValueError(f"FFT2DPropagator only supports 2D input. Got {wave.event_shape}")

        self.add_input("input", wave)
        self._set_generation(wave.generation + 1)
        self.dtype = get_complex_dtype(wave.dtype)
        self.event_shape = wave.event_shape

        out_wave = Wave(event_shape=self.event_shape, batch_size=wave.batch_size, dtype=self.dtype)
        out_wave._set_generation(self._generation + 1)
        out_wave.set_parent(self)
        self.output = out_wave
        return self.output

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"FFT2DProp(gen={gen}, mode={self.precision_mode})"
