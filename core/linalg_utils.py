import numpy as np

def complex_normal_random_array(shape, dtype=np.complex128):
    """
    Generate a complex-valued random array with standard normal distribution
    for both real and imaginary parts.
    """
    real = np.random.randn(*shape)
    imag = np.random.randn(*shape)
    return (real + 1j * imag).astype(dtype)
