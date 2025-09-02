import time
import cProfile, pstats, io
import numpy as np
import cupy as cp
import gpie

def run_with_timer(func, *args, sync_gpu=True, **kwargs):
    """Measure elapsed time for a function, syncing GPU if requested."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if sync_gpu and gpie.core.backend.np().__name__ == "cupy":
        cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start
    return result, elapsed

def profile_with_cprofile(func, *args, sort="cumtime", limit=30, **kwargs):
    """Run cProfile on a function and print top results."""
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s).sort_stats(sort)
    stats.print_stats(limit)
    print(s.getvalue())

def set_backend(backend_name: str):
    """Switch gpie backend between numpy and cupy."""
    import gpie
    if backend_name == "numpy":
        gpie.set_backend(np)
        return np
    elif backend_name == "cupy":
        gpie.set_backend(cp)
        return cp
    else:
        raise ValueError("Backend must be 'numpy' or 'cupy'.")
