import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, replicate
from gpie.core.linalg_utils import random_normal_array
from gpie.core.rng_utils import get_rng


# ------------------------------------------------------------
# CDP model definition
# ------------------------------------------------------------

@model
def coded_diffraction_model(var, masks, damping, dtype=np.complex64):
    """
    Coded diffraction pattern model for scheduling comparison.

    Args:
        var: Noise variance
        masks: ndarray of shape (B, H, W)
        damping: "auto" or float
        dtype: Complex dtype
    """
    B, H, W = masks.shape

    # Prior (single latent image)
    x = ~GaussianPrior(event_shape=(H, W), label="obj", dtype=dtype)

    # Replicate across batch dimension
    x_batch = replicate(x, batch_size=B)

    # Apply masks and FFT
    y = fft2(masks * x_batch)

    # Amplitude measurement
    AmplitudeMeasurement(var=var, damping=damping) << y


# ------------------------------------------------------------
# Single trial runner
# ------------------------------------------------------------

def run_single_trial(
    schedule,
    seed,
    n_iter,
    shape,
    n_measurements,
    noise,
    block_size,
    damping,
):
    """
    Run one CDP reconstruction trial and return PMSE history.
    """
    rng = get_rng(seed)

    # Ground-truth object
    true_obj = random_normal_array(
        (1, *shape),
        dtype=np.complex64,
        rng=rng,
    )

    # Random phase masks
    masks = random_normal_array(
        (n_measurements, *shape),
        dtype=np.complex64,
        rng=rng,
    )

    # Build graph
    g = coded_diffraction_model(
        var=noise,
        masks=masks,
        damping=damping,
        dtype=np.complex64,
    )

    g.set_init_rng(get_rng(seed + 1))

    # Inject true sample and generate observations
    g.get_wave("obj").set_sample(true_obj)
    g.generate_sample(rng=get_rng(seed + 2), update_observed=True)

    pmse_history = []

    def monitor(graph, t):
        est = graph.get_wave("obj").compute_belief().data
        err = pmse(est, true_obj)
        pmse_history.append(err)

    g.run(
        n_iter=n_iter,
        schedule=schedule,
        block_size=block_size,
        callback=monitor,
    )

    return np.array(pmse_history)


# ------------------------------------------------------------
# Experiment runner
# ------------------------------------------------------------

def run_experiment(
    n_trials=10,
    n_iter=200,
    shape=(64, 64),
    n_measurements=4,
    noise=1e-4,
    block_size=1,
    damping="auto",
    output_dir="results",
):
    os.makedirs(output_dir, exist_ok=True)

    schedules = ["parallel", "sequential"]
    histories = {s: [] for s in schedules}

    base_seed = 1000

    for trial in range(n_trials):
        seed = base_seed + trial
        print(f"Trial {trial + 1}/{n_trials} (seed={seed})")

        for schedule in schedules:
            hist = run_single_trial(
                schedule=schedule,
                seed=seed,
                n_iter=n_iter,
                shape=shape,
                n_measurements=n_measurements,
                noise=noise,
                block_size=block_size,
                damping=damping,
            )
            histories[schedule].append(hist)

    # Convert to arrays: shape (n_trials, n_iter)
    for k in histories:
        histories[k] = np.stack(histories[k], axis=0)

    # --------------------------------------------------------
    # Plot statistics
    # --------------------------------------------------------

    iters = np.arange(n_iter)

    plt.figure(figsize=(7, 5))

    for schedule, color in zip(schedules, ["tab:blue", "tab:orange"]):
        data = histories[schedule]

        median = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)

        plt.plot(iters, median, label=schedule, color=color)
        plt.fill_between(iters, q1, q3, color=color, alpha=0.25)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("PMSE")
    plt.title("Sequential vs Parallel Scheduling (CDP)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(output_dir, "sequential_vs_parallel.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved figure to {out_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequential vs Parallel scheduling benchmark (CDP)"
    )

    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--measurements", type=int, default=4)
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--damping", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results")

    args = parser.parse_args()

    try:
        damping_value = float(args.damping)
    except ValueError:
        damping_value = args.damping

    run_experiment(
        n_trials=args.trials,
        n_iter=args.n_iter,
        shape=(args.size, args.size),
        n_measurements=args.measurements,
        noise=args.noise,
        block_size=args.block_size,
        damping=damping_value,
        output_dir=args.output_dir,
    )
