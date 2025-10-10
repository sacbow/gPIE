import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement
from gpie.core.rng_utils import get_rng

from gpie.imaging.ptychography.data.dataset import PtychographyDataset
from gpie.imaging.ptychography.simulator.probe import generate_probe
from gpie.imaging.ptychography.simulator.scan import generate_fermat_spiral_positions

# Utility for loading skimage sample images
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from io_utils import load_sample_image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------------------------
# Model definition
# -----------------------------------------------------------
@model
def ptychography_graph_known_probe(
    obj_shape,
    prb,
    indices,
    noise: float,
    dtype=np.complex64,
    damping: float = 0.3,
):
    """Ptychography factor graph with known probe."""
    obj = ~GaussianPrior(event_shape=obj_shape, label="object", dtype=dtype)
    patches = obj.extract_patches(indices)
    exit_waves = prb * patches
    AmplitudeMeasurement(var=noise, label="meas", damping=damping) << fft2(exit_waves)
    return


# -----------------------------------------------------------
# Build synthetic dataset
# -----------------------------------------------------------
def build_dataset(size=256, noise=1e-4, n_scans=40):
    obj_shape = (size, size)

    amp = load_sample_image("moon", shape=obj_shape)
    phase = load_sample_image("coins", shape=obj_shape)
    obj = amp * np.exp(1j * phase)

    pixel_size = 0.1  # µm / pixel
    prb = generate_probe(
        shape=(128, 128),
        pixel_size=pixel_size,
        aperture_radius=2.0,
        smooth_edge_sigma=0.05,
    )

    dataset = PtychographyDataset()
    dataset.set_pixel_size(pixel_size)
    dataset.set_object(obj)
    dataset.set_probe(prb)

    scan_gen = generate_fermat_spiral_positions(step_um=1.0)
    dataset.simulate_diffraction(scan_gen, max_num_points=n_scans, noise=noise)

    print(f"[INFO] Generated {dataset.size} diffraction patterns.")
    return dataset


# -----------------------------------------------------------
# Run reconstruction
# -----------------------------------------------------------
def run_ptychography(n_iter=100, size=256, noise=1e-4, save_graph=False):
    dataset = build_dataset(size=size, noise=noise, n_scans=40)

    graph = ptychography_graph_known_probe(
        obj_shape=dataset.obj.shape,
        prb=dataset.prb,
        indices=[d.indices for d in dataset],
        noise=noise,
    )

    meas_node = graph.get_factor("meas")
    diffs = [d.diffraction for d in dataset]
    meas_node.set_observed(np.stack(diffs))

    graph.set_init_rng(get_rng(42))
    graph.run(n_iter=n_iter)

    # Retrieve reconstructed object
    recon = graph.get_wave("object").compute_belief().data[0]
    amp = np.abs(recon)
    phase = np.angle(recon)

    # Save reconstructed amplitude/phase
    plt.imsave(f"{RESULTS_DIR}/reconstructed_amp.png", amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/reconstructed_phase.png", phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)

    # Display final result (for interactive use)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    titles = ["Reconstructed Amplitude", "Reconstructed Phase"]
    images = [
        axes[0].imshow(amp, cmap="gray"),
        axes[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi),
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    cbar = fig.colorbar(images[1], ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Phase [rad]", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/reconstructed_composite.png", dpi=200)
    plt.close()

    if save_graph:
        print(f"[INFO] Saving factor graph visualization to {RESULTS_DIR}/graph.html ...")
        graph.visualize(output_path=os.path.join(RESULTS_DIR, "graph.html"))


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ptychography simulation and reconstruction demo with gPIE")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of EP iterations")
    parser.add_argument("--size", type=int, default=256, help="Object size (H=W)")
    parser.add_argument("--noise", type=float, default=1e-4, help="Measurement noise variance")
    parser.add_argument("--save-graph", action="store_true", help="Save factor graph visualization to HTML")
    args = parser.parse_args()

    run_ptychography(n_iter=args.n_iter, size=args.size, noise=args.noise, save_graph=args.save_graph)
