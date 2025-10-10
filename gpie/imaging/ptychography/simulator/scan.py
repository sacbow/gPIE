from typing import Generator, Tuple, Optional
import math
import numpy as np


def _apply_jitter(
    y: float, x: float, jitter_um: float, rng: Optional[np.random.Generator] = None
) -> Tuple[float, float]:
    """Add Gaussian jitter (position noise) to (y, x)."""
    if jitter_um == 0.0:
        return y, x
    rng = rng or np.random.default_rng()
    dy, dx = rng.normal(0.0, jitter_um, size=2)
    return y + dy, x + dx


def generate_raster_positions(
    stride_um: float = 1.0,
    jitter_um: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Generator[Tuple[float, float], None, None]:
    """
    Generator that yields scan positions in real space, starting from (0, 0),
    following a spiral raster pattern (Manhattan spiral).

    Parameters
    ----------
    stride_um : float
        Base spacing between scan positions (μm).
    jitter_um : float, optional
        Standard deviation of Gaussian positional jitter in μm. Default = 0 (no jitter).
    rng : np.random.Generator, optional
        Optional random number generator for reproducibility.

    Yields
    ------
    (y_um, x_um) : tuple of floats
        Real-space scan coordinates with optional jitter.
    """
    yield _apply_jitter(0.0, 0.0, jitter_um, rng)

    step = 1
    y, x = 0, 0

    while True:
        # Move right
        for _ in range(step):
            x += 1
            yield _apply_jitter(y * stride_um, x * stride_um, jitter_um, rng)

        # Move up
        for _ in range(step):
            y += 1
            yield _apply_jitter(y * stride_um, x * stride_um, jitter_um, rng)

        step += 1

        # Move left
        for _ in range(step):
            x -= 1
            yield _apply_jitter(y * stride_um, x * stride_um, jitter_um, rng)

        # Move down
        for _ in range(step):
            y -= 1
            yield _apply_jitter(y * stride_um, x * stride_um, jitter_um, rng)

        step += 1



def generate_fermat_spiral_positions(
    step_um: float = 1.0,
    golden_angle_rad: float = 2.399967,  # ≈ 137.5 deg
    jitter_um: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Generator[Tuple[float, float], None, None]:
    """
    Generator that yields scan positions along a Fermat spiral in real space.

    Parameters
    ----------
    step_um : float
        Radial scaling factor (μm).
    golden_angle_rad : float
        Angle between successive points in radians (default ≈ 137.5°).
    jitter_um : float, optional
        Standard deviation of Gaussian positional jitter in μm. Default = 0 (no jitter).
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Yields
    ------
    (y_um, x_um) : tuple of floats
        Real-space coordinates along Fermat spiral, with optional jitter.
    
    References
    ----------
    X. Huang et al., "Effects of overlap uniformness for ptychography,"
    Opt. Express 22(11), 12634–12644 (2014).
    """
    i = 0
    rng = rng or np.random.default_rng()

    while True:
        r = step_um * math.sqrt(i)
        theta = i * golden_angle_rad
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        yield _apply_jitter(y, x, jitter_um, rng)
        i += 1
