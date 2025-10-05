import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import numpy as np
from examples.ptychography.simulator.probe import generate_probe, make_smooth_random_phase


def test_probe_shape_and_type():
    shape = (128, 128)
    probe = generate_probe(shape=shape, pixel_size=1.0, aperture_radius=0.05)
    assert isinstance(probe, np.ndarray)
    assert probe.shape == shape
    assert np.iscomplexobj(probe)


def test_probe_intensity_normalized():
    probe = generate_probe(shape=(128, 128), pixel_size=1.0, aperture_radius=0.05)
    intensity = np.abs(probe) ** 2
    peak = np.max(intensity)
    assert np.isclose(peak, 1.0, rtol=1e-5)


def test_probe_with_random_phase_changes_output():
    probe1 = generate_probe(shape=(128, 128), pixel_size=1.0, aperture_radius=0.05, random_phase=False)
    probe2 = generate_probe(shape=(128, 128), pixel_size=1.0, aperture_radius=0.05, random_phase=True, seed=42)
    # Expect significant difference due to random phase
    assert not np.allclose(probe1, probe2)


def test_smooth_phase_range():
    phase = make_smooth_random_phase(shape=(128, 128), cutoff_radius=0.03, seed=123)
    assert phase.shape == (128, 128)
    assert np.min(phase) >= 0.0
    assert np.max(phase) <= 2 * np.pi
