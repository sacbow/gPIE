from typing import Tuple, Literal, Optional
from gpie.core.backend import np
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array
from .aperture import make_aperture


def make_smooth_random_phase(
    shape: Tuple[int, int],
    cutoff_radius: float = 0.05,
    seed: int = 0
) -> np().ndarray:
    """
    Generate a smooth random phase map by applying a Gaussian low-pass filter
    to white noise in frequency space.

    Parameters
    ----------
    shape : (ny, nx)
        Output phase map size.
    cutoff_radius : float
        Cutoff frequency for Gaussian low-pass filter (normalized [0, 0.5]).
    seed : int
        RNG seed.

    Returns
    -------
    phase : np.ndarray
        Smooth random phase map (radians in [0, 2π)).
    """
    ny, nx = shape
    fft2, ifft2, fftfreq, fftshift = np().fft.fft2, np().fft.ifft2, np().fft.fftfreq, np().fft.fftshift
    rng = get_rng(seed)

    # Complex Gaussian noise
    noise = random_normal_array(shape=shape, dtype=np().complex64, rng=rng)
    K = fft2(noise)

    # Gaussian low-pass filter
    fy = fftfreq(ny)
    fx = fftfreq(nx)
    FX, FY = np().meshgrid(fx, fy, indexing="ij")
    R = np().sqrt(FX**2 + FY**2)
    gaussian_lpf = np().exp(-(R / cutoff_radius) ** 2)

    # Filter in k-space and transform back
    smoothed = fftshift(ifft2(K * gaussian_lpf))
    phase = np().angle(smoothed)
    return (phase + 2 * np().pi) % (2 * np().pi)


def generate_probe(
    shape: Tuple[int, int],
    pixel_size: float = 1.0,
    aperture_radius: Optional[float] = None,
    *,
    kind: Literal["circular", "square"] = "circular",
    space: Literal["fourier", "real"] = "real",
    smooth_edge_sigma: float = 0.05,
    random_phase: bool = False,
    cutoff_radius: float = 0.03,
    seed: Optional[int] = None,
) -> np().ndarray:
    """
    Generate a soft-aperture probe in real or Fourier space.

    Parameters
    ----------
    shape : (ny, nx)
        Probe shape in pixels.
    pixel_size : float
        Physical pixel size [μm].
    aperture_radius : float
        Aperture radius or half-width (depending on `kind`).
    kind : {"circular", "square"}, optional
        Aperture shape. Default: "circular".
    space : {"fourier", "real"}, optional
        Domain where aperture is applied:
            - "fourier": Apply aperture in Fourier domain, IFFT to real space.
            - "real": Apply aperture directly in real space.
        Default: "fourier".
    smooth_edge_sigma : float, optional
        Gaussian falloff at aperture edge (fraction of radius). Default: 0.1.
    random_phase : bool, optional
        If True, apply smooth random phase modulation. Default: False.
    cutoff_radius : float, optional
        Low-pass cutoff for random phase generator. Default: 0.03.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    probe : complex ndarray
        Real-space probe field (normalized so max amplitude = 1).
    """
    rng = get_rng(seed)
    fft2, ifft2, ifftshift = np().fft.fft2, np().fft.ifft2, np().fft.ifftshift
    if aperture_radius is None:
        aperture_radius = shape[0] * 0.5
    # --- Generate aperture mask ---
    mask = make_aperture(
        shape=shape,
        pixel_size=pixel_size,
        aperture_radius=aperture_radius,
        kind=kind,
        smooth_edge_sigma=smooth_edge_sigma,
    )

    # --- Apply random abberation if requested ---
    if random_phase:
        phase = make_smooth_random_phase(shape, cutoff_radius=cutoff_radius, seed=seed or 0)
        mask = mask * np().exp(1j * phase)

    # --- Domain handling ---
    if space == "fourier":
        probe = ifftshift(ifft2(mask))
    elif space == "real":
        probe = ifftshift(mask.astype(np().complex64))
    else:
        raise ValueError(f"Invalid 'space' argument: {space}. Must be 'fourier' or 'real'.")

    # --- Normalize amplitude ---
    probe /= np().max(np().abs(probe))
    return probe
