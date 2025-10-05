import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple


def make_smooth_random_phase(shape: Tuple[int, int], cutoff_radius: float = 0.05, seed: int = 0) -> np.ndarray:
    """
    Generate a smooth random phase map by applying a Gaussian low-pass filter
    to white noise in k-space, then inverse transforming back.

    Parameters
    ----------
    shape : (ny, nx)
        Output phase map size.
    cutoff_radius : float
        Cutoff frequency for Gaussian low-pass filter (normalized to [0, 0.5]).
    seed : int
        RNG seed.

    Returns
    -------
    phase : np.ndarray
        Smooth random phase map in radians, values in [0, 2π).
    """
    ny, nx = shape
    rng = np.random.default_rng(seed)
    # complex noise
    noise = rng.standard_normal(size=(ny, nx)) + 1j * rng.standard_normal(size=(ny, nx))
    K = fft2(noise)

    # Gaussian low-pass in frequency domain
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    R = np.sqrt(FX**2 + FY**2)
    gaussian_lpf = np.exp(-(R / cutoff_radius) ** 2)

    # Apply filter and go back
    K_filtered = K * gaussian_lpf
    smoothed = fftshift(ifft2(K_filtered))

    phase = np.angle(smoothed)
    phase = (phase + 2 * np.pi) % (2 * np.pi)
    return phase



def generate_probe(
    shape: Tuple[int, int],
    pixel_size: float,
    aperture_radius: float,
    smooth_edge_sigma: float = 0.1,
    random_phase: bool = False,
    cutoff_radius: float = 0.03,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a soft-aperture circular probe in real space.

    Parameters
    ----------
    shape : (ny, nx)
        Output probe shape in pixels.
    pixel_size : float
        Physical pixel size [μm].
    aperture_radius : float
        Radius of aperture in frequency space [1/μm].
    smooth_edge_sigma : float
        Gaussian falloff at edge of aperture [fraction of radius].
    random_phase : bool
        If True, apply weak random phase in k-space.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    probe : complex ndarray
        Real-space probe (normalized so peak intensity = 1).
    """
    ny, nx = shape
    fy = np.fft.fftfreq(ny, d=pixel_size)
    fx = np.fft.fftfreq(nx, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    R = np.sqrt(FX**2 + FY**2)

    # Soft circular aperture in frequency space
    sigma = aperture_radius * smooth_edge_sigma
    mask = np.exp(-0.5 * ((R - aperture_radius) / sigma) ** 2)
    mask[R <= aperture_radius] = 1.0  # inside core radius = 1

    if random_phase:
        phase = make_smooth_random_phase(shape, cutoff_radius=cutoff_radius, seed=seed)
        mask = mask * np.exp(1j * phase)

    # IFFT to real space
    probe = ifftshift(ifft2(mask))
    probe /= np.max(np.abs(probe))  # normalize intensity

    return probe
