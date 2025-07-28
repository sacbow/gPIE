from .backend import get_backend

def _sync_cupy_rng(rng):
    """
    Sync CuPy's global RNG with NumPy rng state for reproducibility.
    Derive a 64-bit integer seed from numpy's bit generator state safely.
    """
    import cupy as cp

    state = rng.bit_generator.state

    if isinstance(state, dict):
        inner_state = state.get("state")
        if isinstance(inner_state, dict):
            seed_candidate = inner_state.get("state")
        else:
            seed_candidate = inner_state
    else:
        seed_candidate = getattr(state, "state", None)

    if callable(seed_candidate):
        seed_candidate = seed_candidate()

    if isinstance(seed_candidate, (list, tuple)):
        seed_val = int(seed_candidate[0])
    elif hasattr(seed_candidate, "__iter__") and not isinstance(seed_candidate, (int, float)):
        seed_val = int(list(seed_candidate)[0])
    elif isinstance(seed_candidate, (int, float)):
        seed_val = int(seed_candidate)
    else:
        import numpy as np
        seed_val = int(np.random.SeedSequence().entropy)

    seed_val = seed_val % (2**63 - 1)
    cp.random.seed(seed_val)



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
        return jax.random.PRNGKey(seed or 0)  # JAX is not yet integrated fully.

    raise NotImplementedError(f"get_rng not implemented for backend '{name}'")


def normal(rng, size, mean=0.0, std=1.0):
    backend = get_backend().__name__

    if backend == "numpy":
        return rng.normal(loc=mean, scale=std, size=size)

    elif backend == "cupy":
        import cupy as cp
        _sync_cupy_rng(rng)
        return std * cp.random.randn(*size) + mean

    else:
        raise NotImplementedError(f"normal() not implemented for backend '{backend}'")


def choice(rng, a, size, replace=True):
    backend = get_backend().__name__

    if backend == "numpy":
        return rng.choice(a, size=size, replace=replace)

    elif backend == "cupy":
        import cupy as cp
        _sync_cupy_rng(rng)
        return cp.random.choice(a, size=size, replace=replace)

    else:
        raise NotImplementedError(f"choice() not implemented for backend '{backend}'")


def shuffle(rng, x):
    backend = get_backend().__name__

    if backend == "numpy":
        return rng.shuffle(x)

    elif backend == "cupy":
        import cupy as cp
        _sync_cupy_rng(rng)
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
        _sync_cupy_rng(rng)
        return cp.random.uniform(low=low, high=high, size=size)

    else:
        raise NotImplementedError(f"uniform() not implemented for backend '{backend}'")
