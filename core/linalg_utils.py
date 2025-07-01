import numpy as np

def reduce_precision_to_scalar(precision_array):
    """
    Reduce a precision array to an equivalent scalar precision
    using harmonic mean of variances.

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
    Generate a complex-valued random array with standard normal distribution.
    Each element follows a circular complex Gaussian distribution with variance 1,
    meaning the real and imaginary parts are N(0, 0.5) independently.

    Args:
        shape (tuple): Shape of the output array.
        dtype (np.dtype): Output complex data type.
        rng (np.random.Generator): Random number generator (required)

    Returns:
        np.ndarray: Complex-valued random array.
    """
    if rng is None:
        raise ValueError("rng must be provided for random sampling.")
    real = rng.normal(loc=0.0, scale=np.sqrt(0.5), size=shape)
    imag = rng.normal(loc=0.0, scale=np.sqrt(0.5), size=shape)
    return (real + 1j * imag).astype(dtype)

def sparse_complex_array(shape, sparsity, dtype=np.complex128, rng=None):
    """
    Generate a sparse complex-valued array where a fraction of entries
    (defined by `sparsity`) are drawn from CN(0,1), and the rest are zero.

    Args:
        shape (tuple): Desired shape of the output array.
        sparsity (float): Fraction (0 < rho <= 1) of non-zero entries.
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