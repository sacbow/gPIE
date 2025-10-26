import argparse
import numpy as np
from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement
from gpie.core.backend import np as gnp
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array
from gpie.imaging.ptychography.simulator.probe import generate_probe
from gpie.imaging.ptychography.simulator.scan import generate_fermat_spiral_positions
from gpie.imaging.ptychography.utils.geometry import (
    realspace_to_pixel_coords,
    filter_positions_within_object,
    slices_from_positions,
)

from benchmark_utils import run_with_timer, profile_with_cprofile, set_backend
from gpie.graph.structure.model import model


# --------------------------------------
# Define the ptychography factor graph
# --------------------------------------
@model
def ptychography_graph_known_probe(obj_shape, prb, indices, noise, dtype=gnp().complex64):
    obj = ~GaussianPrior(event_shape=obj_shape, label="object", dtype=dtype)
    patches = obj.extract_patches(indices)
    exit_waves = prb * patches
    AmplitudeMeasurement(var=noise, label="meas", damping=0.3) << fft2(exit_waves)


def build_ptycho_graph(size=512, noise=1e-4, n_scans=40):
    rng = get_rng(seed=1234)
    obj_shape = (size, size)

    # Generate synthetic object
    obj = random_normal_array((1, *obj_shape), dtype=gnp().complex64, rng=rng)[0]

    # Generate probe
    prb = generate_probe(
        shape=(128, 128),
        pixel_size=0.1,
        aperture_radius=2.0,
        smooth_edge_sigma=0.05,
    )

    # Generate scan positions
    scan_gen = generate_fermat_spiral_positions(step_um=1.0)
    real_positions = [next(scan_gen) for _ in range(n_scans)]
    pix_positions = realspace_to_pixel_coords(real_positions, pixel_size_um=0.1, obj_shape=obj_shape)
    valid_pix_positions = filter_positions_within_object(pix_positions, obj_shape=obj_shape, probe_shape=prb.shape)
    indices = slices_from_positions(valid_pix_positions, probe_shape=prb.shape, obj_shape=obj_shape)

    # Build graph
    g = ptychography_graph_known_probe(
        obj_shape=obj_shape,
        prb=prb,
        indices=indices,
        noise=noise,
    )

    g.set_init_rng(get_rng(123))
    g.generate_sample(rng=rng, update_observed=True)
    return g


def run_ptycho(n_iter=100, size=512, n_scans=40, noise=1e-4, verbose=False):
    g = build_ptycho_graph(size=size, n_scans=n_scans, noise=noise)
    g.run(n_iter=n_iter, callback=None, verbose=False)


# --------------------------------------
# CLI entry point
# --------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Ptychography in gPIE")
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
    parser.add_argument("--size", type=int, default=256,
                        help="Object size (H=W)")
    parser.add_argument("--scans", type=int, default=40,
                        help="Number of scan points")
    parser.add_argument("--noise", type=float, default=1e-4,
                        help="Amplitude noise variance")
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
        profile_with_cprofile(
            run_ptycho,
            n_iter=args.n_iter,
            verbose=args.verbose,
            size=args.size,
            n_scans=args.scans,
            noise=args.noise,
        )
    else:
        _, elapsed = run_with_timer(
            run_ptycho,
            size=args.size,
            n_scans=args.scans,
            n_iter=args.n_iter,
            noise=args.noise,
            verbose=args.verbose,
            sync_gpu=True,
        )
        fft_mode = "fftw" if args.fftw else args.backend
        print(f"[{fft_mode}] Total time: {elapsed:.3f} s")
