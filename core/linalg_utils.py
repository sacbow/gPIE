import numpy as np

def complex_normal_random_array(shape, dtype=np.complex128, seed=None):
    """
    Generate a complex-valued random array with standard normal distribution.
    Each element follows a circular complex Gaussian distribution with variance 1,
    meaning the real and imaginary parts are N(0, 0.5) independently.

    Args:
        shape (tuple): Shape of the output array.
        dtype (np.dtype): Output complex data type.
        seed (int or None): Optional seed for deterministic randomness.

    Returns:
        np.ndarray: Complex-valued random array.
    """
    rng = np.random.default_rng(seed)
    real = rng.normal(loc=0.0, scale=np.sqrt(0.5), size=shape)
    imag = rng.normal(loc=0.0, scale=np.sqrt(0.5), size=shape)
    return (real + 1j * imag).astype(dtype)

def random_unitary_matrix(n, seed=None, dtype=np.complex128):
    """
    Generate a random unitary matrix of shape (n, n) using SVD
    of a circularly-symmetric complex Gaussian matrix.

    Args:
        n (int): Size of the square matrix.
        seed (int or None): Seed for reproducibility.
        dtype (np.dtype): Complex data type (default: complex128).

    Returns:
        np.ndarray: Unitary matrix of shape (n, n).
    """
    A = complex_normal_random_array((n, n), dtype=dtype, seed=seed)
    U, _, _ = np.linalg.svd(A)
    return U


def random_binary_mask(shape, subsampling_rate, seed=None):
    """
    Generate a random boolean mask with given shape and subsampling rate.
    """
    if not (0.0 <= subsampling_rate <= 1.0):
        raise ValueError("subsampling_rate must be between 0.0 and 1.0")

    rng = np.random.default_rng(seed)
    total = np.prod(shape)
    num_true = int(total * subsampling_rate)

    flat_mask = np.zeros(total, dtype=bool)
    flat_mask[:num_true] = True
    rng.shuffle(flat_mask)

    return flat_mask.reshape(shape)

