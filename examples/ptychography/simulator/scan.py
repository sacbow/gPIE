from typing import Generator, Tuple
import math

def generate_raster_positions(
    stride_um: float = 1.0,
) -> Generator[Tuple[float, float], None, None]:
    """
    Generator that yields scan positions in real space, starting from (0, 0),
    following a spiral raster pattern (manhattan spiral).

    The unit of stride is micrometers (μm).

    Yields:
        Tuple[float, float]: (y_um, x_um) positions
    """
    yield (0.0, 0.0)  # start at origin

    step = 1  # step size in grid units
    y, x = 0, 0  # current grid coordinate

    while True:
        # Move right
        for _ in range(step):
            x += 1
            yield (y * stride_um, x * stride_um)

        # Move up
        for _ in range(step):
            y += 1
            yield (y * stride_um, x * stride_um)

        step += 1  # increase ring

        # Move left
        for _ in range(step):
            x -= 1
            yield (y * stride_um, x * stride_um)

        # Move down
        for _ in range(step):
            y -= 1
            yield (y * stride_um, x * stride_um)

        step += 1



def generate_fermat_spiral_positions(
    step_um: float = 1.0,
    golden_angle_rad: float = 2.399967,  # ≈ 137.5 deg
) -> Generator[Tuple[float, float], None, None]:
    """
    Generator that yields scan positions along a Fermat spiral in real space.

    The origin (0, 0) is the center, and the scan expands outward radially.

    Args:
        step_um: Controls radial growth rate (μm).
        golden_angle_rad: Angle between points in radians (default ≈ 137.5 deg)

    Yields:
        Tuple[float, float]: (y_um, x_um) coordinates
    
    References:
        X. Huang et al., "Effects of overlap uniformness for ptychography," Opt. Express 22(11), 12634–12644 (2014).
        https://doi.org/10.1364/OE.22.012634
    """
    i = 0
    while True:
        r = step_um * math.sqrt(i)
        theta = i * golden_angle_rad
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        yield (y, x)
        i += 1


