import argparse
import numpy as np
from numpy.typing import NDArray
from gpie import model, SupportPrior, fft2, AmplitudeMeasurement, pmse
from gpie.core.linalg_utils import circular_aperture, random_phase_mask
from benchmark_utils import run_with_timer, profile_with_cprofile, set_backend
from gpie.core.rng_utils import get_rng


# ==== CDIモデル定義 ====
@model
def random_cdi(support: NDArray[np.bool_], n_layers: int, phase_masks: list[NDArray], var: float):
    x = ~SupportPrior(support=support, label="sample", dtype=np.complex64)
    for i in range(n_layers):
        x = fft2(phase_masks[i] * x)
    AmplitudeMeasurement(var=var, damping=0.3) << x


def build_random_cdi_graph(H=1024, W=1024, var=1e-4, support_radius=0.3, n_layers=2):
    """Structured Random Matrix CDIモデルのGraph構築"""
    rng = get_rng(seed=42)
    shape = (H, W)
    support = circular_aperture(shape, radius=support_radius)
    phase_masks = [random_phase_mask(shape, rng=rng, dtype=np.complex64) for _ in range(n_layers)]

    g = random_cdi(support=support, n_layers=n_layers, phase_masks=phase_masks, var=var)
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=999), update_observed=True)
    return g


def run_random_cdi(n_iter=100, verbose=False):
    g = build_random_cdi_graph()
    true_x = g.get_wave("sample").get_sample()

    def monitor(graph, t):
        if verbose and t % 10 == 0:
            est_x = graph.get_wave("sample").compute_belief().data
            err = pmse(est_x, true_x)
            print(f"[t={t}] PMSE = {err:.5e}")

    g.run(n_iter=n_iter, callback=monitor, verbose=False)


# ==== CLIエントリーポイント ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Structured Random Matrix CDI in gPIE")
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
    args = parser.parse_args()

    backend = set_backend(
        backend_name=args.backend,
        use_fftw=args.fftw,
        threads=args.threads,
        planner_effort=args.planner_effort,
    )

    if args.profile:
        profile_with_cprofile(run_random_cdi, n_iter=args.n_iter, verbose=args.verbose)
    else:
        _, elapsed = run_with_timer(
            run_random_cdi, n_iter=args.n_iter, verbose=args.verbose, sync_gpu=True
        )
        fft_mode = "fftw" if args.fftw else args.backend
        print(f"[{fft_mode}] Total time: {elapsed:.3f} s")
