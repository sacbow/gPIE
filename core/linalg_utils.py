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

