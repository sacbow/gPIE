import argparse
from benchmark_utils import run_with_timer, profile_with_cprofile, set_backend
from gpie import model, SupportPrior, fft2, AmplitudeMeasurement, mse
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import circular_aperture, masked_random_array
import numpy as np


@model
def holography(support, var, ref_wave):
    obj = ~SupportPrior(support=support, label="obj", dtype=np.complex64)
    AmplitudeMeasurement(var=var) << (fft2(ref_wave + obj))


def build_holography_graph(H=1024, W=1024, noise=1e-4):
    rng = get_rng(seed=42)
    support_x = circular_aperture((H, W), radius=0.2, center=(-0.2, -0.2))
    data_x = masked_random_array(support_x, dtype=np.complex64, rng=rng)
    support_y = circular_aperture((H, W), radius=0.2, center=(0.2, 0.2))
    g = holography(support=support_y, var=noise, ref_wave=data_x)
    g.set_init_rng(get_rng(11))
    g.generate_sample(rng=get_rng(9), update_observed=True)
    return g


def run_holography(n_iter=100, verbose=False):
    g = build_holography_graph()
    true_obj = g.get_wave("obj").get_sample()

    def monitor(graph, t):
        if verbose and t % 10 == 0:
            err = mse(graph.get_wave("obj").compute_belief().data, true_obj)
            print(f"[t={t}] PSE (sum)={err:.5e}")

    g.run(n_iter=n_iter, callback=monitor, verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark holography in gPIE")
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
                        help="Print progress and PSE during iterations")
    args = parser.parse_args()

    backend = set_backend(
        backend_name=args.backend,
        use_fftw=args.fftw,
        threads=args.threads,
        planner_effort=args.planner_effort,
    )

    if args.profile:
        profile_with_cprofile(run_holography, n_iter=args.n_iter, verbose=args.verbose)
    else:
        _, elapsed = run_with_timer(
            run_holography, n_iter=args.n_iter, verbose=args.verbose, sync_gpu=True
        )
        fft_mode = "fftw" if args.fftw else args.backend
        print(f"[{fft_mode}] Total time: {elapsed:.3f} s")
