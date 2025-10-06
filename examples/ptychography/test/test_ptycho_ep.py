# test/test_ptycho_ep.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import pytest
import numpy as np
from examples.ptychography.data.dataset import PtychographyDataset
from examples.ptychography.simulator.scan import generate_raster_positions
from examples.ptychography.simulator.probe import generate_probe
from examples.ptychography.algorithms.ptycho_ep import PtychoEP


@pytest.fixture
def small_dataset():
    """Generate a small synthetic dataset for EP reconstruction tests."""
    rng = np.random.default_rng(0)
    ds = PtychographyDataset()
    ds.set_pixel_size(1.0)

    # Object: 32×32 complex field
    obj = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    prb = generate_probe(
        shape=(16, 16),
        pixel_size=1.0,
        aperture_radius=0.1,
        random_phase=False,
        seed=0,
    )
    ds.set_object(obj.astype(np.complex64))
    ds.set_probe(prb.astype(np.complex64))

    # Single scan point
    scan_gen = generate_raster_positions(stride_um=2.0)
    ds.simulate_diffraction(scan_gen, max_num_points=1, noise=1e-6, rng=rng)
    return ds


def test_ep_initialization(small_dataset):
    """Ensure EP engine initializes correctly and builds valid graph."""
    engine = PtychoEP(dataset=small_dataset, noise=1e-6, damping=0.1)
    g = engine.graph

    # Basic graph structure
    assert hasattr(g, "get_factor"), "Graph should implement get_factor()"
    assert g.get_factor("meas") is not None, "Measurement factor not found."

    # Object and probe nodes exist
    obj_wave = g.get_wave("object")
    prb_wave = g.get_wave("probe")
    assert obj_wave is not None and prb_wave is not None

    # Dtypes consistent
    assert obj_wave.dtype == engine.dtype
    assert isinstance(engine.noise, float)


def test_ep_run_executes(small_dataset):
    """Ensure EP reconstruction runs without errors and produces valid samples."""
    engine = PtychoEP(dataset=small_dataset, noise=1e-6, damping=0.5)
    g = engine.run(n_iter=5, verbose=False)

    # Check that inference completed and object/probe estimates exist
    obj_rec = engine.get_object()
    prb_rec = engine.get_probe()

    assert obj_rec.shape == small_dataset.obj.shape
    assert prb_rec.shape == small_dataset.prb.shape

    assert np.all(np.isfinite(obj_rec)), "NaN detected in reconstructed object."
    assert np.all(np.isfinite(prb_rec)), "NaN detected in reconstructed probe."


def test_ep_determinism(small_dataset):
    """EP reconstruction should be deterministic for a fixed seed."""
    engine1 = PtychoEP(dataset=small_dataset, noise=1e-6, damping=0.0, seed=123)
    engine2 = PtychoEP(dataset=small_dataset, noise=1e-6, damping=0.0, seed=123)

    g1 = engine1.run(n_iter=3, verbose=False)
    g2 = engine2.run(n_iter=3, verbose=False)

    obj1, obj2 = engine1.get_object(), engine2.get_object()
    np.testing.assert_allclose(obj1, obj2, rtol=1e-5, atol=1e-5)
