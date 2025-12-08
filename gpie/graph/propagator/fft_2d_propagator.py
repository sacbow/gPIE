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

    def compute_belief(self):
        """
        Compute joint belief over (x, y) under the unitary mapping y = F x.

        This uses the current messages:
            - msg_x: message on input wave x
            - msg_y: message on output wave y

        and updates:
            - self.x_belief
            - self.y_belief
        """
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required to compute belief.")

        # Ensure complex dtype compatibility
        if not np().issubdtype(msg_x.data.dtype, np().complexfloating):
            msg_x = msg_x.astype(self.dtype)

        mode = self._precision_mode

        if mode == UnaryPropagatorPrecisionMode.SCALAR:
            # All scalar precision: belief_x is product in x-domain,
            # then transform to y-domain using FFT.
            self.x_belief = msg_x * msg_y.ifft2_centered()
            self.y_belief = self.x_belief.fft2_centered()

        elif mode == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY:
            # Input scalar precision, output array precision.
            self.y_belief = msg_x.fft2_centered().as_array_precision() * msg_y
            self.x_belief = self.y_belief.ifft2_centered()

        elif mode == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR:
            # Input array precision, output scalar precision.
            self.x_belief = msg_x * msg_y.ifft2_centered().as_array_precision()
            self.y_belief = self.x_belief.fft2_centered()

        else:
            raise ValueError(f"Unknown precision_mode: {self._precision_mode}")

    def forward(self):
        """
        EP-style forward pass: propagate a message from x to y.

        Initial iteration:
            - If there is no output message and no cached belief yet,
              send y-message as FFT of the x-message.

        Steady-state:
            - Use cached y_belief to compute the new message to y via
              m_y = y_belief / (current outgoing message).
        """
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        out_msg = self.output_message
        yb = self.y_belief

        # Initial iteration: no outgoing message and no belief yet
        if out_msg is None and yb is None:
            if msg_x is None:
                raise RuntimeError(
                    "FFT2DPropagator.forward(): missing input message on the initial iteration. "
                    "Upstream prior must emit an initial message before FFT."
                )

            # Apply FFT to propagate from x-domain to y-domain
            msg = msg_x.fft2_centered()

            # Align precision mode with the output wave
            if self.output.precision_mode_enum == PrecisionMode.ARRAY:
                msg = msg.as_array_precision()

            self.output.receive_message(self, msg)
            return

        # Steady-state EP update using cached belief
        if out_msg is not None and yb is not None:
            msg = yb / out_msg
            self.output.receive_message(self, msg)
            return

        # Inconsistent internal state
        raise RuntimeError(
            "FFT2DPropagator.forward(): inconsistent state. "
            "Expected both y_belief and output_message to be None (initial) or both present (update)."
        )

    def backward(self):
        """
        EP-style backward pass: propagate a message from y back to x.

        Uses the current output message and cached beliefs to compute:
            m_x = x_belief / (incoming message from other factors).
        """
        x_wave = self.inputs["input"]
        if self.output_message is None:
            raise RuntimeError("Output message missing.")

        # Update beliefs based on current messages
        self.compute_belief()
        incoming = self.input_messages[x_wave]
        msg = self.x_belief / incoming
        x_wave.receive_message(self, msg)

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
