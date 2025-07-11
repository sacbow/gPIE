import numpy as np
import warnings

def reduce_precision_to_scalar(precision_array):
    """
    Reduce a precision array to an equivalent scalar precision
    using harmonic mean of variances.

    This is used to summarize spatially varying precision into a single scalar value,
    often needed when combining messages in approximate inference.

    Args:
        precision_array (np.ndarray): Elementwise precision (positive)

    Returns:
        float: Scalar precision value
    """
    precision_array = np.asarray(precision_array, dtype=np.float64)
    if np.any(precision_array <= 0):
        raise ValueError("Precision values must be positive.")
    return 1.0 / np.mean(1.0 / precision_array)


def complex_normal_random_array(shape, dtype=np.complex128, rng=None):
    """
    (Deprecated) Generate complex Gaussian random array from CN(0,1).

    This function is retained for backward compatibility. Use random_normal_array instead.

    Args:
        shape (tuple): Output shape.
        dtype (np.dtype): Complex dtype (default: complex128)
        rng (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Complex-valued array
    """
    warnings.warn(
        "complex_normal_random_array is deprecated. Use random_normal_array instead.",
        DeprecationWarning
    )
    rng = np.random.default_rng() if rng is None else rng
    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    return (real + 1j * imag).astype(dtype)


def random_normal_array(shape, dtype=np.complex128, rng=None):
    """
    Generate random array from standard normal distribution with given dtype.

    Supports both real and complex outputs depending on dtype:
        - real: N(0,1)
        - complex: CN(0,1), i.e., real and imag ~ N(0,1)

    Args:
        shape (tuple): Output shape.
        dtype (np.dtype): Desired dtype (real or complex).
        rng (np.random.Generator or None): Random generator.

    Returns:
        np.ndarray: Random array with specified dtype.
    """
    rng = np.random.default_rng() if rng is None else rng
    if np.issubdtype(dtype, np.complexfloating):
        real = rng.normal(size=shape)
        imag = rng.normal(size=shape)
        return (real + 1j * imag).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        return rng.normal(size=shape).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def sparse_complex_array(shape, sparsity, dtype=np.complex128, rng=None):
    """
    Generate a sparse complex-valued array where a fraction of entries
    (defined by `sparsity`) are drawn from CN(0,1), and the rest are zero.

    Args:
        shape (tuple): Desired shape of the output array.
        sparsity (float): Fraction (0 < sparsity <= 1) of non-zero entries.
        dtype (np.dtype): Complex data type (default: complex128).
        rng (np.random.Generator): Random number generator.

    Returns:
        np.ndarray: Complex array with sparse non-zero entries.
    """
    if rng is None:
        raise ValueError("rng must be provided for sparse_complex_array.")

    N = np.prod(shape)
    num_nonzero = int(sparsity * N)

    sample = np.zeros(N, dtype=dtype)
    idx = rng.choice(N, size=num_nonzero, replace=False)
    real = rng.normal(0.0, np.sqrt(0.5), num_nonzero)
    imag = rng.normal(0.0, np.sqrt(0.5), num_nonzero)
    sample[idx] = real + 1j * imag

    return sample.reshape(shape)


def random_unitary_matrix(n, dtype=np.complex128, rng=None):
    """
    Generate a random unitary matrix of shape (n, n) using SVD
    of a circularly-symmetric complex Gaussian matrix.

    Useful for constructing sensing matrices or unitary propagators.

    Args:
        n (int): Size of the square matrix.
        dtype (np.dtype): Complex data type (default: complex128).
        rng (np.random.Generator): Random number generator (required)

    Returns:
        np.ndarray: Unitary matrix of shape (n, n).
    """
    A = complex_normal_random_array((n, n), dtype=dtype, rng=rng)
    U, _, _ = np.linalg.svd(A)
    return U


def random_binary_mask(shape, subsampling_rate, rng=None):
    """
    Generate a random boolean mask with given shape and subsampling rate.

    Args:
        shape (tuple): Shape of the mask.
        subsampling_rate (float): Ratio of True entries (between 0 and 1).
        rng (np.random.Generator): Random number generator (required)

    Returns:
        np.ndarray: Boolean mask array.
    """
    if not (0.0 <= subsampling_rate <= 1.0):
        raise ValueError("subsampling_rate must be between 0.0 and 1.0")
    if rng is None:
        raise ValueError("rng must be provided for random sampling.")

    total = np.prod(shape)
    num_true = int(total * subsampling_rate)
    flat_mask = np.zeros(total, dtype=bool)
    flat_mask[:num_true] = True
    rng.shuffle(flat_mask)
    return flat_mask.reshape(shape)


def random_phase_mask(shape, dtype=np.complex128, rng=None):
    """
    Generate a complex-valued random phase mask with unit magnitude.

    Each entry is drawn as exp(1j * theta) with theta ~ Uniform[0, 2π].

    Args:
        shape (tuple): Shape of the mask.
        dtype (np.dtype): Complex dtype.
        rng (np.random.Generator or None): Random generator.

    Returns:
        ndarray: Complex array with unit modulus (e.g., exp(1j * theta)).
    """
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.uniform(0, 2 * np.pi, size=shape)
    return np.exp(1j * theta).astype(dtype)

def circular_aperture(shape, radius, center=None):
    """
    Generate a circular binary mask (True inside the circle).

    Args:
        shape (tuple): 2D shape (H, W)
        radius (float): Normalized radius (0 < r < 0.5), min_dim = 1
        center (tuple or None): Normalized coordinates (cy, cx), origin at center, unit=min_dim

    Returns:
        np.ndarray: Boolean mask with circular region True
    """
    H, W = shape
    if not (0.0 < radius < 0.5):
        raise ValueError("radius must be between 0 and 0.5")

    min_dim = min(H, W)
    abs_radius = radius * min_dim

    cy_pix = H // 2
    cx_pix = W // 2
    if center is not None:
        dx = center[0] * min_dim
        dy = -center[1] * min_dim  
        cx_pix = int(round(W // 2 + dx))  
        cy_pix = int(round(H // 2 + dy))  

    if not (0 <= cy_pix < H and 0 <= cx_pix < W):
        raise ValueError("center out of bounds")

    yy, xx = np.ogrid[:H, :W]
    dist2 = (yy - cy_pix) ** 2 + (xx - cx_pix) ** 2
    return dist2 <= abs_radius ** 2


def square_aperture(shape, radius, center=None):
    """
    Generate a square binary mask (True inside the square).

    Args:
        shape (tuple): 2D shape (H, W)
        radius (float): Normalized half-side (0 < r < 0.5), min_dim = 1
        center (tuple or None): Normalized coordinates (cy, cx), origin at center, unit=min_dim

    Returns:
        np.ndarray: Boolean mask with square region True
    """
    H, W = shape
    if not (0.0 < radius < 0.5):
        raise ValueError("radius must be between 0 and 0.5")

    min_dim = min(H, W)
    half = int(radius * min_dim)

    cy_pix = H // 2
    cx_pix = W // 2
    if center is not None:
        dx = center[0] * min_dim
        dy = -center[1] * min_dim  
        cx_pix = int(round(W // 2 + dx))  
        cy_pix = int(round(H // 2 + dy))  


    y0, y1 = cy_pix - half, cy_pix + half
    x0, x1 = cx_pix - half, cx_pix + half

    if y0 < 0 or y1 >= H or x0 < 0 or x1 >= W:
        raise ValueError("square aperture goes out of bounds")

    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask

def fft2_centered(x: np.ndarray) -> np.ndarray:
    """
    Centered 2D FFT (with fftshift and ifftshift).
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))

def ifft2_centered(x: np.ndarray) -> np.ndarray:
    """
    Centered 2D inverse FFT (with fftshift and ifftshift).
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), norm="ortho"))