import numpy as np
import matplotlib.pyplot as plt
from typing import List
from gpie.imaging.ptychography.data.diffraction_data import DiffractionData


def plot_scan_positions(positions: List[tuple], ax=None):
    """Plot scan positions in real space (μm)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    y, x = zip(*positions)
    ax.scatter(x, y, c="tab:blue", s=30, edgecolor="k")
    ax.set_aspect("equal")
    ax.set_xlabel("x [μm]")
    ax.set_ylabel("y [μm]")
    ax.set_title("Scan positions")
    ax.grid(True, ls=":", color="gray", alpha=0.5)

    return fig



def plot_diffraction_patterns(diff_data_list: List[DiffractionData], ncols=4, log_scale=True):
    """Plot a grid of diffraction patterns."""
    n = len(diff_data_list)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols , 3 * nrows + 3))
    axes = np.array(axes).ravel()

    for i, (d, ax) in enumerate(zip(diff_data_list, axes)):
        img = np.abs(d.diffraction)
        if log_scale:
            img = np.log10(img + 1e-8)
        ax.imshow(img, cmap="jet")
        ax.set_title(f"{i}: pos=({d.position[0]:.1f}, {d.position[1]:.1f})")
        ax.axis("off")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Diffraction patterns", fontsize=20)
    fig.tight_layout()
    return fig


def plot_object_and_probe(obj: np.ndarray, prb: np.ndarray):
    """Visualize object and probe amplitude/phase."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    amp_obj = np.abs(obj)
    phs_obj = np.angle(obj)
    amp_prb = np.abs(prb)
    phs_prb = np.angle(prb)

    im0 = axes[0, 0].imshow(amp_obj, cmap="gray")
    axes[0, 0].set_title("Object amplitude")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(phs_obj, cmap="twilight")
    axes[0, 1].set_title("Object phase")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(amp_prb, cmap="gray")
    axes[1, 0].set_title("Probe amplitude")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(phs_prb, cmap="twilight")
    axes[1, 1].set_title("Probe phase")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle("Object and Probe", fontsize=14)
    fig.tight_layout()
    return fig
