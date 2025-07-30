import warnings
from .backend import get_backend

def _sync_cupy_rng(rng):
    """
    Sync CuPy's global RNG with NumPy rng state for reproducibility.
    """
    try:
        import cupy as cp
    except ImportError:
        return

    # Only sync if NumPy RNG is passed
    if hasattr(rng, "bit_generator"):
        state = rng.bit_generator.state
        if isinstance(state, dict):
            inner_state = state.get("state")
            seed_candidate = inner_state.get("state") if isinstance(inner_state, dict) else inner_state
        else:
            seed_candidate = getattr(state, "state", None)
        if isinstance(seed_candidate, (list, tuple)):
            seed_val = int(seed_candidate[0])
        elif isinstance(seed_candidate, int):
            seed_val = seed_candidate
        else:
            import numpy as np
            seed_val = int(np.random.SeedSequence().entropy)
        cp.random.seed(seed_val % (2**63 - 1))


def get_rng(seed=None):
    """Return an RNG appropriate for the current backend."""
    backend_name = get_backend().__name__

    import numpy as np
    if backend_name == "numpy":
        return np.random.default_rng(seed)

    elif backend_name == "cupy":
        try:
            import cupy as cp
            return cp.random.default_rng(seed)
        except ImportError:
            warnings.warn("CuPy backend selected but CuPy not installed. Falling back to NumPy RNG.")
            return np.random.default_rng(seed)

    elif "jax" in backend_name:
        import jax
        return jax.random.PRNGKey(seed or 0)

    raise NotImplementedError(f"get_rng not implemented for backend '{backend_name}'")


def _ensure_rng_backend(rng):
    """If RNG backend mismatches, issue warning and fallback to current backend's RNG."""
    backend_name = get_backend().__name__
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if backend_name == "numpy" and not isinstance(rng, np.random.Generator):
        warnings.warn("[rng_utils] RNG backend mismatch, replacing with numpy RNG.")
        return np.random.default_rng()
    elif backend_name == "cupy" and cp and not isinstance(rng, cp.random.Generator):
        warnings.warn("[rng_utils] RNG backend mismatch, replacing with cupy RNG.")
        return cp.random.default_rng()

    return rng


def normal(rng, size, mean=0.0, std=1.0):
    rng = _ensure_rng_backend(rng)
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if isinstance(rng, np.random.Generator):
        return rng.normal(loc=mean, scale=std, size=size)
    elif cp and isinstance(rng, cp.random.Generator):
        size = (size,) if isinstance(size, int) else size
        return std * rng.standard_normal(size) + mean


def choice(rng, a, size, replace=True):
    rng = _ensure_rng_backend(rng)
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if isinstance(rng, np.random.Generator):
        return rng.choice(a, size=size, replace=replace)
    elif cp and isinstance(rng, cp.random.Generator):
        return cp.random.choice(a, size=size, replace=replace)


def shuffle(rng, x):
    rng = _ensure_rng_backend(rng)
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if isinstance(rng, np.random.Generator):
        rng.shuffle(x)
        return x
    elif cp and isinstance(rng, cp.random.Generator):
        cp.random.shuffle(x)
        return x


def uniform(rng, low=0.0, high=1.0, size=None):
    rng = _ensure_rng_backend(rng)
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = None

    if isinstance(rng, np.random.Generator):
        return rng.uniform(low=low, high=high, size=size)
    elif cp and isinstance(rng, cp.random.Generator):
        size = (size,) if isinstance(size, int) else size
        return rng.uniform(low=low, high=high, size=size)
