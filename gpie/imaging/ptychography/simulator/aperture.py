from typing import Tuple, Literal
from gpie.core.backend import np


def make_aperture(
    shape: Tuple[int, int],
    pixel_size: float,
    aperture_radius: float,
    kind: Literal["circular", "square"] = "circular",
    smooth_edge_sigma: float = 0.01,
) -> np().ndarray:
    """
    Generate an aperture mask (circular or square) in the frequency or real space.

    Parameters
    ----------
    shape : (ny, nx)
        Size of the aperture (pixels).
    pixel_size : float
        Pixel size [μm].
    aperture_radius : float
        Radius (for circular) or half-width (for square) in 1/μm (frequency domain) or μm (real space).
    kind : {"circular", "square"}
        Aperture shape.
    smooth_edge_sigma : float
        Gaussian falloff (as fraction of radius or half-width).

    Returns
    -------
    mask : np.ndarray
        Aperture mask, values in [0, 1].
    """
    ny, nx = shape
    fy = np().fft.fftfreq(ny, d=pixel_size)
    fx = np().fft.fftfreq(nx, d=pixel_size)
    FX, FY = np().meshgrid(fx, fy, indexing="ij")

    if kind == "circular":
        R = np().sqrt(FX**2 + FY**2)
        sigma = aperture_radius * smooth_edge_sigma
        mask = np().exp(-0.5 * ((R - aperture_radius) / sigma) ** 2)
        mask[R <= aperture_radius] = 1.0

    elif kind == "square":
        # half-width along each axis
        wx = wy = aperture_radius
        smooth_x = np().exp(-0.5 * ((np().abs(FX) - wx) / (wx * smooth_edge_sigma)) ** 2)
        smooth_y = np().exp(-0.5 * ((np().abs(FY) - wy) / (wy * smooth_edge_sigma)) ** 2)
        mask = np().minimum(smooth_x, smooth_y)
        mask[(np().abs(FX) <= wx) & (np().abs(FY) <= wy)] = 1.0
    else:
        raise ValueError(f"Unknown aperture kind: {kind}")

    return mask
