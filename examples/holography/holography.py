# examples/holography/holography.py

import os
import numpy as np
import matplotlib.pyplot as plt

from gpie import Graph, SupportPrior, fft2, AmplitudeMeasurement, mse
from gpie.core.linalg_utils import circular_aperture, masked_random_array

# Create result directory
os.makedirs("examples/holography/results", exist_ok=True)


class HolographyGraph(Graph):
    """
    A factor graph model for inline holography using EP.
    The graph consists of:
    - A complex object with known support
    - A reference wave
    - FFT-based forward model
    - Amplitude-only measurement
    """
    def __init__(self, var, ref_wave, support):
        super().__init__()
        obj = ~SupportPrior(support=support, label="obj", dtype=np.complex64)
        with self.observe():
            AmplitudeMeasurement(var=var) @ (fft2(ref_wave + obj))
        self.compile()


def build_holography_graph(H=512, W=512, noise=1e-4):
    """
    Construct the holography graph with a circular object and reference wave.
    """
    rng = np.random.default_rng(seed=42)

    # Create reference wave (data_x) on the left
    support_x = circular_aperture((H, W), radius=0.2, center=(-0.2, -0.2))
    data_x = masked_random_array(support_x, dtype=np.complex64, rng=rng)

    # Create object support on the right
    support_y = circular_aperture((H, W), radius=0.2, center=(0.2, 0.2))

    # Construct the graph
    g = HolographyGraph(var=noise, ref_wave=data_x, support=support_y)
    g.set_init_rng(np.random.default_rng(11))
    g.generate_sample(rng=np.random.default_rng(9), update_observed=True)
    return g


def run_holography(n_iter=100):
    """
    Run EP inference for the holography problem and save results.
    """
    g = build_holography_graph()
    obj_wave = g.get_wave("obj")
    true_obj = obj_wave.get_sample()

    pse_list = []

    def monitor(graph, t):
        if t % 5 == 0 or t == n_iter - 1:
            est = graph.get_wave("obj").compute_belief().data
            err = mse(est, true_obj)
            pse_list.append(err)
            print(f"[t={t}] PSE = {err:.5e}")

    g.run(n_iter=n_iter, callback=monitor)

    # Get final estimate
    est = obj_wave.compute_belief().data

    # Save amplitude and phase images
    amp = np.abs(est)
    phase = np.angle(est)

    plt.imsave("examples/holography/results/reconstructed_amp.png", amp, cmap="gray")
    plt.imsave("examples/holography/results/reconstructed_phase.png", phase, cmap="twilight")

    # Save convergence curve
    plt.figure()
    plt.plot(np.arange(0, n_iter, 5), pse_list, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("PSE (MSE)")
    plt.title("Convergence Curve")
    plt.grid(True)
    plt.savefig("examples/holography/results/convergence.png")
    plt.close()


if __name__ == "__main__":
    run_holography(n_iter=100)
