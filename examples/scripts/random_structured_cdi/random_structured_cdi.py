from pathlib import Path

readme_path = Path("random_structured_cdi.py")

import argparse
import numpy as np
import matplotlib.pyplot as plt

from gpie import model, SupportPrior, fft2, AmplitudeMeasurement, pmse
from gpie.core.linalg_utils import circular_aperture, random_phase_mask

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from io_utils import load_sample_image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@model
def random_structured_cdi(support, n_layers, phase_masks, var):
    x = ~SupportPrior(support=support, label="sample", dtype=np.complex64)
    for i in range(n_layers):
        x = fft2(phase_masks[i] * x)
    AmplitudeMeasurement(var=var, damping=0.4) << x


def build_random_cdi_graph(H=256, W=256, var=1e-4, support_radius=0.3, n_layers=2):
    rng = np.random.default_rng(seed=42)
    shape = (H, W)
    support = circular_aperture(shape, radius=support_radius)

    # Load object image and apply support mask
    amp = load_sample_image("camera", shape=shape)
    phase = load_sample_image("moon", shape=shape)
    obj = amp * np.exp(1j * 2 * np.pi * phase)
    obj *= support

    phase_masks = [random_phase_mask(shape, rng=rng, dtype=np.complex64) for _ in range(n_layers)]
    g = random_structured_cdi(support = support, n_layers = n_layers, phase_masks = phase_masks, var = var)
    g.set_init_rng(np.random.default_rng(seed=1))
    g.get_wave("sample").set_sample(obj)
    g.generate_sample(rng=np.random.default_rng(seed=999), update_observed=True)
    return g, obj


def run_random_cdi(n_iter=100, size=256, n_layers=2, support_radius=0.3, save_graph=False):
    g, true_obj = build_random_cdi_graph(H=size, W=size, n_layers=n_layers, support_radius=support_radius)
    pse_list = []

    def monitor(graph, t):
        if t % 10 == 0 or t == n_iter - 1:
            est = graph.get_wave("sample").compute_belief().data
            err = pmse(est, true_obj)
            pse_list.append(err)
            print(f"[t={t}] PMSE = {err:.5e}")

    g.run(n_iter=n_iter, callback=monitor)

    est = g.get_wave("sample").compute_belief().data[0]
    amp = np.abs(est)
    phase = np.angle(est) * (np.abs(true_obj) > 1e-5)

    plt.imsave(f"{RESULTS_DIR}/reconstructed_amp.png", amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/reconstructed_phase.png", phase, cmap="twilight")

    true_amp = np.abs(true_obj)
    true_phase = np.angle(true_obj) * (true_amp > 1e-5)

    plt.imsave(f"{RESULTS_DIR}/true_amp.png", true_amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/true_phase.png", true_phase, cmap="twilight")

    plt.figure()
    plt.plot(np.arange(0, len(pse_list) * 10, 10), pse_list, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("PMSE")
    plt.yscale('log')
    plt.title("Convergence Curve")
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/convergence.png")
    plt.close()

    if save_graph:
        print(f"Saving factor graph visualization to {RESULTS_DIR}/graph.html ...")
        g.visualize(output_path=os.path.join(RESULTS_DIR, "graph.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Structured CDI demo with gPIE")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of EP iterations")
    parser.add_argument("--size", type=int, default=256, help="Image size (H=W)")
    parser.add_argument("--layers", type=int, default=2, help="Number of random modulation layers")
    parser.add_argument("--support-radius", type=float, default=0.3, help="Radius of support mask")
    parser.add_argument("--save-graph", action="store_true", help="Save factor graph visualization")
    args = parser.parse_args()

    run_random_cdi(n_iter=args.n_iter,
                   size=args.size,
                   n_layers=args.layers,
                   support_radius=args.support_radius,
                   save_graph=args.save_graph)

