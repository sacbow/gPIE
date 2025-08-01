import argparse
from benchmark_utils import run_with_timer, profile_with_cprofile, set_backend
from gpie import Graph, SupportPrior, fft2, AmplitudeMeasurement, mse
from gpie.core.linalg_utils import circular_aperture, masked_random_array
import numpy as np

class HolographyGraph(Graph):
    def __init__(self, var, ref_wave, support):
        super().__init__()
        obj = ~SupportPrior(support=support, label="obj", dtype=np.complex64)
        with self.observe():
            _ = AmplitudeMeasurement(var=var) @ (fft2(ref_wave + obj))
        self.compile()

def build_holography_graph(H=512, W=512, noise=1e-4):
    rng = np.random.default_rng(seed=42)
    support_x = circular_aperture((H,W), radius=0.2, center=(-0.2, -0.2))
    data_x = masked_random_array(support_x, dtype=np.complex64, rng=rng)
    support_y = circular_aperture((H,W), radius=0.2, center=(0.2, 0.2))
    g = HolographyGraph(var=noise, ref_wave=data_x, support=support_y)
    g.set_init_rng(np.random.default_rng(11))
    g.generate_sample(rng=np.random.default_rng(9), update_observed=True)
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
                        help="Backend to use (numpy or cupy)")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="Number of EP iterations")
    parser.add_argument("--profile", action="store_true",
                        help="Enable cProfile profiling")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress and PSE during iterations")
    args = parser.parse_args()

    backend = set_backend(args.backend)

    if args.profile:
        profile_with_cprofile(run_holography, n_iter=args.n_iter, verbose=args.verbose)
    else:
        _, elapsed = run_with_timer(run_holography, n_iter=args.n_iter, verbose=args.verbose, sync_gpu=True)
        print(f"[{args.backend}] Total time: {elapsed:.3f} s")
