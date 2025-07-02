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
