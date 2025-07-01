import numpy as np
from .base import Propagator
from core.uncertain_array import UncertainArray as UA
from graph.wave import Wave

class FFT2DPropagator(Propagator):
    """
    A 2D Fourier Transform-based propagator for use in graphical models.
    This class applies a centered 2D FFT (via fftshift + fft2 + ifftshift)
    and its inverse for belief propagation in the frequency domain.

    This is typically used for imaging problems such as ptychography,
    phase retrieval, or any model involving forward FFT transforms.
    """
    def __init__(self, shape, dtype=np.complex128):
        """
        Initialize the FFT2DPropagator.

        Args:
            shape (tuple): Shape of the 2D input/output array.
            dtype (np.dtype): Complex dtype (default: np.complex128).
        """
        super().__init__(input_names=("input",), dtype=dtype)
        self.shape = shape
        self._init_rng = None
        self.x_belief = None
        self.y_belief = None

    def set_init_rng(self, rng):
        """
        Set the random number generator used for message initialization.
        """
        self._init_rng = rng

    def _fft2_centered(self, x):
        """
        Compute centered 2D FFT: shift → fft2 → shift back.

        Args:
            x (ndarray): Spatial domain input.

        Returns:
            ndarray: Frequency domain output with DC centered.
        """
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

    def _ifft2_centered(self, y):
        """
        Compute centered 2D inverse FFT: shift → ifft2 → shift back.

        Args:
            y (ndarray): Frequency domain input.

        Returns:
            ndarray: Spatial domain output with origin centered.
        """
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(y), norm="ortho"))

    def compute_belief(self):
        """
        Compute forward and backward beliefs using centered FFT and IFFT.
        This combines incoming messages from the spatial and frequency domains.
        """
        x_wave = self.inputs["input"]
        msg_x = self.input_messages.get(x_wave)
        msg_y = self.output_message

        if msg_x is None or msg_y is None:
            raise RuntimeError("Both input and output messages are required.")

        # Extract data and precision
        r = msg_x.data
        gamma = msg_x.to_scalar_precision()
        p = msg_y.data
        tau = msg_y.precision

        # Forward: apply FFT and fuse messages
        Ur = self._fft2_centered(r)
        denom = gamma + tau
        y_mean = (gamma / denom) * Ur + (tau / denom) * p
        scalar_prec = 1.0 / np.mean(1.0 / denom)

        self.y_belief = UA(y_mean, dtype=self.dtype, precision=scalar_prec)

        # Backward: apply IFFT to get belief in input domain
        x_mean = self._ifft2_centered(y_mean)
        self.x_belief = UA(x_mean, dtype=self.dtype, precision=scalar_prec)

    def forward(self):
        """
        Send a message from this factor to the output wave.

        If the output message is not yet initialized, generate a random one.
        Otherwise, use the current belief to compute the updated message.
        """
        if self.output_message is None or self.y_belief is None:
            if self._init_rng is None:
                raise RuntimeError("Initial RNG not configured.")
            msg = UA.random(self.shape, dtype=self.dtype, rng=self._init_rng)
        else:
            msg = self.y_belief / self.output_message

        self.output.receive_message(self, msg)

    def backward(self):
        """
        Send a message from this factor to the input wave.
        This requires computing belief from the output side.
        """
        x_wave = self.inputs["input"]
        if self.output_message is None:
            raise RuntimeError("Output message missing.")

        self.compute_belief()
        msg_in = self.x_belief / UA.as_scalar_precision(self.input_messages[x_wave])
        x_wave.receive_message(self, msg_in)

    def __matmul__(self, wave: Wave) -> Wave:
        """
        Overload the @ operator to connect input wave to this propagator.

        Args:
            wave (Wave): Input wave object (must be 2D).

        Returns:
            Wave: Output wave connected to this propagator.
        """
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
        """
        Generate output sample y = F x from input sample x using FFT.

        Args:
            rng (np.random.Generator): RNG (not used here but required by interface).
        """
        x_wave = self.inputs["input"]
        x = x_wave.get_sample()
        if x is None:
            raise RuntimeError("Input sample not set.")
        y = self._fft2_centered(x)
        self.output.set_sample(y)

    def __repr__(self):
        gen = self._generation if self._generation is not None else "-"
        return f"FFT2DProp(gen={gen})"
