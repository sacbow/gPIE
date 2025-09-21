import argparse
import numpy as np
import matplotlib.pyplot as plt

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse
from gpie.core.linalg_utils import random_phase_mask
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from io_utils import load_sample_image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


@model
def coded_diffraction_pattern(shape, n_measurements, phase_masks, noise):
    x = ~GaussianPrior(event_shape=shape, label="sample", dtype = np.complex64)
    for i in range(n_measurements):
        y = phase_masks[i] * x
        z = fft2(y)
        AmplitudeMeasurement(var=noise, damping=0.3) << z


def build_cdp_graph(H=256, W=256, noise=1e-4, n_measurements=4):
    rng = np.random.default_rng(seed=42)
    shape = (H, W)
    phase_masks = [random_phase_mask(shape, rng=rng, dtype=np.complex64) for _ in range(n_measurements)]

    g = coded_diffraction_pattern(shape = shape, n_measurements = n_measurements, phase_masks = phase_masks, noise = noise)
    g.set_init_rng(np.random.default_rng(seed=1))

    amp = load_sample_image("camera", shape=shape)
    phase = load_sample_image("moon", shape=shape)
    complex_img = amp * np.exp(1j * 2 * np.pi * phase)
    g.get_wave("sample").set_sample(complex_img.astype(np.complex64))

    g.generate_sample(rng=np.random.default_rng(seed=999), update_observed=True)
    return g


def run_cdp(n_iter=100, size=256, n_measurements=4, save_graph=False):
    g = build_cdp_graph(H=size, W=size, n_measurements=n_measurements)
    true_x = g.get_wave("sample").get_sample()

    pse_list = []

    def monitor(graph, t):
        if t % 10 == 0 or t == n_iter - 1:
            est = graph.get_wave("sample").compute_belief().data
            err = pmse(est, true_x)
            pse_list.append(err)
            print(f"[t={t}] PMSE = {err:.5e}")

    g.run(n_iter=n_iter, callback=monitor)

    est = g.get_wave("sample").compute_belief().data[0]
    amp = np.abs(est)
    phase = np.angle(est) * (np.abs(true_x[0]) > 1e-5)

    plt.imsave(f"{RESULTS_DIR}/reconstructed_amp.png", amp, cmap="gray")
    plt.imsave(f"{RESULTS_DIR}/reconstructed_phase.png", phase, cmap="twilight")

    true_amp = np.abs(true_x[0])
    true_phase = np.angle(true_x[0]) * (true_amp > 1e-5)
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
    parser = argparse.ArgumentParser(description="Coded Diffraction Pattern demo with gPIE")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of EP iterations")
    parser.add_argument("--size", type=int, default=256, help="Image size (H=W)")
    parser.add_argument("--measurements", type=int, default=4, help="Number of phase masks")
    parser.add_argument("--save-graph", action="store_true", help="Save factor graph visualization")
    args = parser.parse_args()

    run_cdp(n_iter=args.n_iter,
            size=args.size,
            n_measurements=args.measurements,
            save_graph=args.save_graph)