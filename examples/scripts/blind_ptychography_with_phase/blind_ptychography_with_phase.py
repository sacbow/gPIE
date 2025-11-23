#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Blind ptychography demo with phase observation (bilinear reconstruction)

This script demonstrates a simplified ptychography model in which
both the object and probe are unknown, but the complex-valued diffraction
fields (including phase) are observed. The goal is to reconstruct both
latent fields under the Expectation Propagation (EP) inference framework.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from gpie import model, fft2, GaussianPrior, GaussianMeasurement
from gpie.core.rng_utils import get_rng
from gpie.imaging.ptychography.simulator.probe import generate_probe
from gpie.imaging.ptychography.simulator.scan import generate_fermat_spiral_positions
from gpie.imaging.ptychography.utils.geometry import realspace_to_pixel_coords, slices_from_positions

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
def blind_ptychography_with_phase_observation(
    obj_shape,
    prb_shape,
    indices,
    noise: float,
    dtype=np.complex64,
):
    """Blind ptychography factor graph with phase observation."""
    obj = ~GaussianPrior(event_shape=obj_shape, label="object", dtype=dtype)
    prb = ~GaussianPrior(event_shape=prb_shape, label="probe", dtype=dtype)
    patches = obj.extract_patches(indices)
    exit_waves = prb * patches
    GaussianMeasurement(var=noise, label="meas") << fft2(exit_waves)
    return


# -----------------------------------------------------------
# Build synthetic dataset
# -----------------------------------------------------------
def build_dataset(size=256, noise=1e-3, n_scans=50):
    """Generate synthetic object, probe, and complex-valued diffraction data."""
    obj_shape = (size, size)

    # --- Ground truth object ---
    amp = load_sample_image("coins", shape=obj_shape)
    phase = load_sample_image("moon", shape=obj_shape)
    obj = amp * np.exp(1j * phase)

    # --- Probe generation ---
    pixel_size = 0.1  # µm per pixel
    prb_shape = (128, 128)
    prb = generate_probe(
        shape=prb_shape,
        pixel_size=pixel_size,
        space="fourier",
        aperture_radius=0.3,
        smooth_edge_sigma=0.1,
    )

    # --- Scan positions ---
    scan_gen = generate_fermat_spiral_positions(step_um=0.5)
    positions_real = [next(scan_gen) for _ in range(n_scans)]
    positions_pixel = realspace_to_pixel_coords(
        positions_real=positions_real,
        pixel_size_um=pixel_size,
        obj_shape=obj_shape,
    )
    indices = slices_from_positions(pixel_positions = positions_pixel,
                                probe_shape = prb_shape,
                                obj_shape = obj_shape)

    return obj, prb, indices, pixel_size


# -----------------------------------------------------------
# Run reconstruction
# -----------------------------------------------------------
def run_blind_ptychography(n_iter=100, size=256, noise=1e-3, save_graph=False):
    obj_gt, prb_gt, indices, pixel_size = build_dataset(size=size, noise=noise, n_scans=50)

    # Build factor graph
    graph = blind_ptychography_with_phase_observation(
        obj_shape=obj_gt.shape,
        prb_shape=prb_gt.shape,
        indices=indices,
        noise=noise,
    )

    # Generate synthetic measurements
    graph.set_init_rng(get_rng(42))
    graph.get_wave("object").set_sample(obj_gt)
    graph.get_wave("probe").set_sample(prb_gt)
    graph.generate_sample(rng=get_rng(999), update_observed=True)

    print("[INFO] Starting EP reconstruction ...")
    graph.run(n_iter=n_iter)

    # Retrieve beliefs
    obj_belief = graph.get_wave("object").compute_belief()
    prb_belief = graph.get_wave("probe").compute_belief()
    obj_est, prb_est = obj_belief.data[0], prb_belief.data[0]
    obj_prec, prb_prec = obj_belief.precision()[0], prb_belief.precision()[0]

    # -----------------------------------------------------------
    # Visualization and Save Results
    # -----------------------------------------------------------
    def save_img(filename, data, cmap="gray", vmin=None, vmax=None):
        plt.imsave(os.path.join(RESULTS_DIR, filename), data, cmap=cmap, vmin=vmin, vmax=vmax)

    # Object
    save_img("object_amplitude.png", np.abs(obj_est), cmap="gray")
    save_img("object_phase.png", np.angle(obj_est), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    save_img("object_precision.png", np.log10(obj_prec + 1e-12), cmap="inferno")

    # Probe
    save_img("probe_amplitude.png", np.abs(prb_est), cmap="jet")
    save_img("probe_phase.png", np.angle(prb_est), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    save_img("probe_precision.png", np.log10(prb_prec + 1e-12), cmap="inferno")

    print(f"[INFO] Saved reconstructed images to {RESULTS_DIR}/")

    # Composite visualization (4-column layout)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    imgs = [
        axes[0].imshow(np.abs(obj_est), cmap="gray"),
        axes[1].imshow(np.log10(obj_prec + 1e-12), cmap="inferno"),
        axes[2].imshow(np.abs(prb_est), cmap="viridis"),
        axes[3].imshow(np.log10(prb_prec + 1e-12), cmap="inferno"),
    ]
    titles = [
        "Object Amplitude",
        "Object Precision (log₁₀)",
        "Probe Amplitude",
        "Probe Precision (log₁₀)",
    ]
    for ax, title, im in zip(axes, titles, imgs):
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "blind_ptychography_summary.png"), dpi=200)
    plt.close()
    print(f"[INFO] Saved summary figure to {RESULTS_DIR}/blind_ptychography_summary.png")

    if save_graph:
        graph.visualize(output_path=os.path.join(RESULTS_DIR, "graph.html"))
        print(f"[INFO] Factor graph visualization saved to {RESULTS_DIR}/graph.html")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blind ptychography demo with phase observation (bilinear reconstruction)")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of EP iterations")
    parser.add_argument("--size", type=int, default=256, help="Object size (H=W)")
    parser.add_argument("--noise", type=float, default=1e-3, help="Measurement noise variance")
    parser.add_argument("--save-graph", action="store_true", help="Save factor graph visualization to HTML")
    args = parser.parse_args()

    run_blind_ptychography(n_iter=args.n_iter, size=args.size, noise=args.noise, save_graph=args.save_graph)
