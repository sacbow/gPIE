import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# test/test_simulation.py
import pytest
import numpy as np
from examples.ptychography.data.dataset import PtychographyDataset
from examples.ptychography.data.diffraction_data import DiffractionData
from examples.ptychography.simulator.scan import generate_raster_positions


def test_simulate_diffraction_single_patch():
    """
    Test that simulate_diffraction generates the correct diffraction pattern
    for a single scan position, by comparing against direct FFT computation.

    Workflow:
        1. Create random object and probe.
        2. Run simulate_diffraction with a single scan position.
        3. Compute the expected diffraction pattern via np.fft2.
        4. Compare both results elementwise.
    """
    rng = np.random.default_rng(42)
    ds = PtychographyDataset()
    ds.set_pixel_size(1.0)

    # --- Step 1: Define object and probe ---
    obj = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    prb = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    ds.set_object(obj)
    ds.set_probe(prb)

    # --- Step 2: Raster scan generator (first point only) ---
    scan_gen = generate_raster_positions(stride_um=1.0)

    # --- Step 3: Run simulate_diffraction for one scan point ---
    ds.simulate_diffraction(
        scan_generator=scan_gen,
        max_num_points=1,    # single patch
        noise=1e-10,
        rng=rng,
    )

    assert len(ds) == 1, "Only one diffraction pattern should be generated."
    diff_data: DiffractionData = ds[0]

    # --- Step 4: Compute expected diffraction pattern manually ---
    # indices defines the region of the object used for this scan
    idx = diff_data.indices
    obj_patch = obj[idx]
    expected_exit_wave = obj_patch * prb

    # orthonormal FFT with centered coordinates
    expected_diffraction = np.abs(np.fft.fftshift(
                                        np.fft.fft2(
                                            np.fft.ifftshift(expected_exit_wave),
                                            norm="ortho"
                                        )))

    # --- Step 5: Compare generated and expected patterns ---
    sim_diffraction = diff_data.diffraction

    np.testing.assert_allclose(
        sim_diffraction,
        expected_diffraction,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Simulated diffraction does not match expected FFT result."
    )
