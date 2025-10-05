import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
import pytest

from examples.ptychography.data.dataset import PtychographyDataset
from examples.ptychography.utils.geometry import slices_from_positions
from examples.ptychography.simulator.forward import generate_diffraction_data
from examples.ptychography.data.diffraction_data import DiffractionData
from examples.ptychography.utils.geometry import realspace_to_pixel_coords, filter_positions_within_object

def test_generate_diffraction_data_basic():
    # Setup dummy object and probe
    obj = np.ones((64, 64), dtype=np.complex64)
    prb = np.ones((16, 16), dtype=np.complex64)
    pixel_size = 0.5  # μm/px

    dataset = PtychographyDataset()
    dataset.set_object(obj)
    dataset.set_probe(prb)
    dataset.set_pixel_size(pixel_size)

    # Define scan positions (real space, μm)
    positions_real = [(0.0, 0.0), (2.0, -2.0), (-2.0, 2.0)]  # μm

    # Convert to pixel coordinates and slice indices
    pixel_coords = realspace_to_pixel_coords(positions_real, pixel_size, obj.shape)
    filtered_pixel_coords = filter_positions_within_object(pixel_coords, obj.shape, prb.shape)
    filtered_real_positions = [pos for pos, pix in zip(positions_real, pixel_coords) if pix in filtered_pixel_coords]
    indices = slices_from_positions(filtered_pixel_coords, prb.shape, obj.shape)

    # Generate diffraction data
    noise_var = 1e-4
    result = generate_diffraction_data(dataset, indices, filtered_real_positions, noise=noise_var)

    # Checks
    assert isinstance(result, list)
    assert all(isinstance(d, DiffractionData) for d in result)
    assert len(result) == len(indices)

    for d in result:
        # Check shape matches probe
        assert d.diffraction.shape == prb.shape
        # Position is float-valued real-space coordinate
        assert isinstance(d.position[0], float) and isinstance(d.position[1], float)
        # Indices are present
        assert isinstance(d.indices, tuple) and len(d.indices) == 2
        # Noise is set correctly
        assert d.noise == pytest.approx(noise_var)


def test_generate_diffraction_matches_manual_fft():
    # RNG for reproducibility
    rng = np.random.default_rng(seed=42)

    # Simulate object and probe
    obj = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
    prb = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    pixel_size = 1.0  # μm/px

    dataset = PtychographyDataset()
    dataset.set_object(obj.astype(np.complex64))
    dataset.set_probe(prb.astype(np.complex64))
    dataset.set_pixel_size(pixel_size)

    # Scan position at center
    positions_real = [(0.0, 0.0)]
    pixel_coords = realspace_to_pixel_coords(positions_real, pixel_size, obj.shape)
    filtered_coords = filter_positions_within_object(pixel_coords, obj.shape, prb.shape)
    assert len(filtered_coords) == 1
    indices = slices_from_positions(filtered_coords, prb.shape, obj.shape)

    # Run simulation with zero noise
    diffs = generate_diffraction_data(
        dataset,
        indices=indices,
        positions_real=positions_real,
        noise=0.0,
        rng=rng,
    )

    # Ground truth calculation (NumPy)
    s0, s1 = indices[0]
    obj_patch = obj[s0, s1]
    exit_wave = obj_patch * prb
    diff_expected = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(exit_wave), norm = "ortho")))

    # Compare to simulated diffraction
    diff_simulated = np.abs(diffs[0].diffraction)

    assert diff_expected.shape == diff_simulated.shape
    assert np.allclose(diff_simulated, diff_expected, rtol=1e-5, atol=1e-6)