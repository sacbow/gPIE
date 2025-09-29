import argparse
from gpie.core.rng_utils import get_rng
import numpy as np
from numpy.typing import NDArray
from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, replicate
from gpie.core.linalg_utils import random_phase_mask
from benchmark_utils import run_with_timer, profile_with_cprofile, set_backend


# ==== CDI definition (fork-based) ====
@model
def coded_diffraction_pattern(noise: float, n_measurements: int, phase_masks: np.ndarray, shape: tuple[int, int]):
    """
    Fork-based coded diffraction pattern model.
    phase_masks: ndarray with shape (B, H, W)
    """
    H, W = shape
    B = n_measurements

    # single object prior
    x = ~GaussianPrior(event_shape=shape, label="sample", dtype=np.complex64)

    # replicate across batch
    x_batch = replicate(x, batch_size=B)

    # apply masks (batched)
    y = fft2(phase_masks * x_batch)

    # single amplitude measurement (batched)
    AmplitudeMeasurement(var=noise, damping=0.3) << y


def build_cdp_graph(H=1024, W=1024, noise=1e-4, n_measurements=4):
    rng = get_rng(seed=42)
    shape = (H, W)
    # batched random phase masks
    phase_masks = random_phase_mask((n_measurements, *shape), rng=rng, dtype=np.complex64)

    g = coded_diffraction_pattern(noise=noise, n_measurements=n_measurements,
                                  phase_masks=phase_masks, shape=shape)
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=999), update_observed=True)
    return g


def run_cdp(n_iter=100, verbose=False):
    g = build_cdp_graph()
    true_x = g.get_wave("sample").get_sample()

    def monitor(graph, t):
        if verbose and t % 10 == 0:
            est_x = graph.get_wave("sample").compute_belief().data
            err = pmse(est_x, true_x)
            print(f"[t={t}] PSE (sum) = {err:.5e}")

    g.run(n_iter=n_iter, callback=monitor, verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Coded Diffraction Pattern (CDP) in gPIE")
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy",
                        help="Numerical backend to use (numpy or cupy)")
    parser.add_argument("--fftw", action="store_true",
                        help="Use FFTW backend (only valid with --backend numpy)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of FFTW threads (only used with --fftw)")
    parser.add_argument("--planner-effort", type=str, default="FFTW_ESTIMATE",
                        help="FFTW planner effort (only used with --fftw)")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="Number of EP iterations")
    parser.add_argument("--profile", action="store_true",
                        help="Enable cProfile profiling")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress and PMSE during iterations")
    parser.add_argument("--size", type=int, default=1024,
                        help="Image size (H=W)")
    parser.add_argument("--measurements", type=int, default=4,
                        help="Number of coded diffraction measurements")
    args = parser.parse_args()

    backend = set_backend(
        backend_name=args.backend,
        use_fftw=args.fftw,
        threads=args.threads,
        planner_effort=args.planner_effort,
    )

    if args.profile:
        profile_with_cprofile(run_cdp, n_iter=args.n_iter, verbose=args.verbose)
    else:
        _, elapsed = run_with_timer(
            run_cdp, n_iter=args.n_iter, verbose=args.verbose, sync_gpu=True
        )
        fft_mode = "fftw" if args.fftw else args.backend
        print(f"[{fft_mode}] Total time: {elapsed:.3f} s")
