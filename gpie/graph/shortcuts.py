from .wave import Wave
from .propagator.fft_2d_propagator import FFT2DPropagator
from .propagator.ifft_2d_propagator import IFFT2DPropagator
from .propagator.fork_propagator import ForkPropagator


def fft2(x: Wave, *, label: str = None) -> Wave:
    """
    Apply 2D centered FFT via FFT2DPropagator.
    """
    if len(x.event_shape) != 2:
        raise ValueError(f"fft2 expects 2D input, got ndim={len(x.event_shape)}")
    y = FFT2DPropagator(dtype=x.dtype) @ x
    if label:
        y.set_label(label)
    return y


def ifft2(x: Wave, *, label: str = None) -> Wave:
    """
    Apply 2D centered inverse FFT via IFFT2DPropagator.
    """
    if len(x.event_shape) != 2:
        raise ValueError(f"ifft2 expects 2D input, got ndim={len(x.event_shape)}")
    y = IFFT2DPropagator(dtype=x.dtype) @ x
    if label:
        y.set_label(label)
    return y


def replicate(x: Wave, batch_size: int, *, label: str = None) -> Wave:
    """
    Replicate a single-sample Wave into a batched Wave via ForkPropagator.

    Args:
        x (Wave): Input wave (must have batch_size=1).
        batch_size (int): Number of replicas to create.
        label (str, optional): Optional label for output wave.

    Returns:
        Wave: Replicated wave with batch_size=batch_size.
    """
    if x.batch_size != 1:
        raise ValueError(f"replicate expects input with batch_size=1, got {x.batch_size}")
    y = ForkPropagator(batch_size=batch_size, dtype=x.dtype) @ x
    if label:
        y.set_label(label)
    return y
