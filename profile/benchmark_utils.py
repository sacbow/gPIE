import time
import cProfile, pstats, io
import warnings
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


def set_backend(
    backend_name: str,
    use_fftw: bool = False,
    threads: int = 1,
    planner_effort: str = "FFTW_ESTIMATE",
):
    """
    Switch gpie backend between numpy, cupy, and optionally fftw.

    Args:
        backend_name (str): Numerical backend ("numpy" or "cupy").
        use_fftw (bool): If True, use FFTW backend (only valid with numpy).
        threads (int): Number of FFTW threads (default=1).
        planner_effort (str): FFTW planner effort, e.g. "FFTW_ESTIMATE", "FFTW_MEASURE".
                              Default: "FFTW_ESTIMATE".

    Returns:
        The active backend module (numpy or cupy).
    """
    from gpie.core import fft as gpie_fft

    # --- set numerical backend ---
    if backend_name == "numpy":
        gpie.set_backend(np)
    elif backend_name == "cupy":
        gpie.set_backend(cp)
    else:
        raise ValueError("Backend must be 'numpy' or 'cupy'.")

    # --- handle fftw flag ---
    if use_fftw:
        if backend_name == "cupy":
            raise RuntimeError("FFTW backend cannot be used with CuPy numerical backend.")
        gpie_fft.set_fft_backend("fftw", threads=threads, planner_effort=planner_effort)
    else:
        # default FFT backend (numpy or cupy)
        gpie_fft.set_fft_backend(backend_name)

        # warn if fftw-specific args were provided without --fftw
        if threads != 1 or planner_effort != "FFTW_ESTIMATE":
            warnings.warn(
                "--threads/--planner-effort ignored because --fftw was not specified",
                UserWarning,
            )

    return gpie.get_backend()
