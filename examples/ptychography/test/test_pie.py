# test/test_pie.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import pytest
import numpy as np
from examples.ptychography.data.dataset import PtychographyDataset
from examples.ptychography.simulator.scan import generate_raster_positions
from examples.ptychography.algorithms.pie import PIE


@pytest.fixture
def small_dataset():
    """Generate a tiny synthetic dataset for PIE tests."""
    rng = np.random.default_rng(0)
    ds = PtychographyDataset()
    ds.set_pixel_size(1.0)

    # Object: 16×16 complex array
    obj = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    prb = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    ds.set_object(obj.astype(np.complex64))
    ds.set_probe(prb.astype(np.complex64))

    # Single scan point
    scan_gen = generate_raster_positions(stride_um=1.0)
    ds.simulate_diffraction(scan_gen, max_num_points=1, noise=1e-6, rng=rng)
    return ds


def test_pie_initialization(small_dataset):
    """Ensure PIE initializes correctly with given dataset."""
    engine = PIE(dataset=small_dataset, alpha=0.1)
    assert engine.obj.shape == small_dataset.obj.shape
    assert engine.prb.shape == small_dataset.prb.shape
    assert engine.dtype in (np.complex64, np.complex128)


def test_pie_run_executes(small_dataset):
    """Ensure PIE run() executes without errors and returns valid object."""
    engine = PIE(dataset=small_dataset, alpha=0.1)
    obj_before = engine.obj.copy()
    obj_after = engine.run(n_iter=3)

    assert obj_after.shape == engine.obj.shape
    assert np.all(np.isfinite(obj_after)), "NaN or inf detected in reconstructed object."

    # The object should change slightly (but not explode)
    diff_norm = np.linalg.norm(obj_after - obj_before)
    assert diff_norm > 0.0, "PIE did not update object at all."
    assert diff_norm < 1e3, "PIE updates diverged abnormally."


def test_pie_deterministic_behavior(small_dataset):
    """Ensure PIE is deterministic for a fixed seed."""
    engine1 = PIE(dataset=small_dataset, alpha=0.05, seed=42)
    engine2 = PIE(dataset=small_dataset, alpha=0.05, seed=42)
    out1 = engine1.run(n_iter=2)
    out2 = engine2.run(n_iter=2)
    np.testing.assert_allclose(out1, out2, rtol=1e-5, atol=1e-5)
