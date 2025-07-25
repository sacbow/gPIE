from .backend import get_backend

def get_rng(seed=None):
    backend = get_backend()
    name = backend.__name__

    if name == "numpy":
        import numpy as np
        return np.random.default_rng(seed)

    elif name == "cupy":
        import cupy as cp
        return cp.random.default_rng(seed)

    elif "jax" in name:
        import jax
        return jax.random.PRNGKey(seed or 0)  # JAXは要特殊処理

    raise NotImplementedError(f"get_rng not implemented for backend '{name}'")

def normal(rng, size, mean=0.0, std=1.0):
    backend = get_backend().__name__

    if backend == "numpy":
        return rng.normal(loc=mean, scale=std, size=size)

    elif backend == "cupy":
        import cupy as cp
        return std * cp.random.randn(*size) + mean

    else:
        raise NotImplementedError(f"normal() not implemented for backend '{backend}'")

def choice(rng, a, size, replace=True):
    backend = get_backend().__name__

    if backend == "numpy":
        return rng.choice(a, size=size, replace=replace)

    elif backend == "cupy":
        import cupy as cp
        return cp.random.choice(a, size=size, replace=replace)

    else:
        raise NotImplementedError(f"choice() not implemented for backend '{backend}'")

def shuffle(rng, x):
    backend = get_backend().__name__

    if backend == "numpy":
        return rng.shuffle(x)

    elif backend == "cupy":
        import cupy as cp
        cp.random.shuffle(x)
        return x

    else:
        raise NotImplementedError(f"shuffle() not implemented for backend '{backend}'")

def uniform(rng, low=0.0, high=1.0, size=None):
    backend = get_backend().__name__

    if backend == "numpy":
        return rng.uniform(low=low, high=high, size=size)

    elif backend == "cupy":
        import cupy as cp
        return cp.random.uniform(low=low, high=high, size=size)

    else:
        raise NotImplementedError(f"uniform() not implemented for backend '{backend}'")

