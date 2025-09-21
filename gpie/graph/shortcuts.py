from .wave import Wave
from .propagator.fft_2d_propagator import FFT2DPropagator
from .propagator.ifft_2d_propagator import IFFT2DPropagator

def fft2(x: Wave, *, label: str = None) -> Wave:
    """
    Apply 2D centered FFT via FFT2DPropagator.

    Args:
        x (Wave): 2D wave in spatial domain.
        label (str, optional): Optional label for output wave.

    Returns:
        Wave: Wave in frequency domain.
    """
    if len(x.event_shape) != 2:
        raise ValueError(f"fft2 expects 2D input, got ndim={x.ndim}")
    y = FFT2DPropagator(dtype=x.dtype) @ x
    if label:
        y.set_label(label)
    return y


def ifft2(x: Wave, *, label: str = None) -> Wave:
    """
    Apply 2D centered inverse FFT via IFFT2DPropagator.

    Args:
        x (Wave): 2D wave in frequency domain.
        label (str, optional): Optional label for output wave.

    Returns:
        Wave: Wave in spatial domain.
    """
    if len(x.event_shape) != 2:
        raise ValueError(f"ifft2 expects 2D input, got ndim={x.ndim}")
    y = IFFT2DPropagator(dtype=x.dtype) @ x
    if label:
        y.set_label(label)
    return y
