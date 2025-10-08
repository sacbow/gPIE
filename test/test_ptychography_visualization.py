import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # disable GUI backend for headless test environments

from gpie.imaging.ptychography.data.dataset import PtychographyDataset
from gpie.imaging.ptychography.simulator.scan import generate_raster_positions

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples")))
from io_utils import load_sample_image


@pytest.fixture(scope="module")
def sample_dataset():
    """Generate a small synthetic ptychography dataset for visualization tests."""
    ds = PtychographyDataset()
    ds.set_pixel_size(1.0)

    # Object & probe
    obj = load_sample_image("camera", shape=(64, 64))
    obj = obj * np.exp(1j * np.zeros_like(obj))
    prb = np.ones((16, 16), dtype=np.complex64)

    ds.set_object(obj.astype(np.complex64))
    ds.set_probe(prb)

    # Simulate a few diffraction patterns
    gen = generate_raster_positions(stride_um=4.0)
    ds.simulate_diffraction(gen, max_num_points=4, noise=1e-6)
    return ds


def test_load_sample_image(tmp_path):
    """Verify that sample image loads and is normalized to [0,1]."""
    img = load_sample_image("camera", shape=(64, 64), save_dir=tmp_path)
    assert img.shape == (64, 64)
    assert np.all((0.0 <= img) & (img <= 1.0)), "Image must be normalized to [0, 1]."


def test_show_object_and_probe(sample_dataset):
    """Ensure that object/probe visualization runs without errors."""
    ds = sample_dataset
    fig = ds.show_object_and_probe()
    assert hasattr(fig, "savefig"), "Expected matplotlib Figure."
    fig.clf()  # release memory


def test_show_scan_positions(sample_dataset):
    """Ensure that scan position visualization runs."""
    ds = sample_dataset
    fig = ds.show_scan_positions()
    assert hasattr(fig, "savefig"), "Expected matplotlib Figure."
    fig.clf()


def test_show_diffraction_patterns(sample_dataset):
    """Ensure that diffraction pattern visualization runs."""
    ds = sample_dataset
    fig = ds.show_diffraction_patterns(ncols=2)
    assert hasattr(fig, "savefig"), "Expected matplotlib Figure."
    fig.clf()
