import pytest
import numpy as np
from gpie.imaging.ptychography.simulator.probe import generate_probe, make_smooth_random_phase


# ---------- Basic properties ----------

def test_probe_shape_and_type_default():
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


# ---------- Random phase ----------

def test_probe_with_random_phase_changes_output():
    probe1 = generate_probe(shape=(128, 128), pixel_size=1.0, aperture_radius=0.05, random_phase=False)
    probe2 = generate_probe(shape=(128, 128), pixel_size=1.0, aperture_radius=0.05, random_phase=True, seed=42)
    # Expect significant difference due to random phase
    assert not np.allclose(probe1, probe2)


def test_random_phase_reproducibility():
    p1 = generate_probe(shape=(64, 64), pixel_size=1.0, aperture_radius=0.05, random_phase=True, seed=123)
    p2 = generate_probe(shape=(64, 64), pixel_size=1.0, aperture_radius=0.05, random_phase=True, seed=123)
    # Same seed → identical probe
    assert np.allclose(p1, p2)


# ---------- Smooth random phase map ----------

def test_smooth_phase_range():
    phase = make_smooth_random_phase(shape=(128, 128), cutoff_radius=0.03, seed=123)
    assert phase.shape == (128, 128)
    assert np.min(phase) >= 0.0
    assert np.max(phase) <= 2 * np.pi


# ---------- New functionality: kind and space ----------

@pytest.mark.parametrize("kind", ["circular", "square"])
@pytest.mark.parametrize("space", ["fourier", "real"])
def test_probe_variants_kind_space(kind, space):
    """Ensure probes are generated correctly for all combinations of aperture type and domain."""
    probe = generate_probe(
        shape=(64, 64),
        pixel_size=1.0,
        aperture_radius=0.05,
        kind=kind,
        space=space,
    )
    assert probe.shape == (64, 64)
    assert np.iscomplexobj(probe)
    assert np.isclose(np.max(np.abs(probe)), 1.0, rtol=1e-5)


def test_invalid_space_raises_error():
    """Invalid space argument should raise ValueError."""
    with pytest.raises(ValueError):
        generate_probe((32, 32), pixel_size=1.0, aperture_radius=0.05, space="invalid")


def test_square_and_circular_differ():
    """Square and circular apertures should yield different intensity patterns."""
    p1 = generate_probe((64, 64), 1.0, 0.05, kind="circular", space="fourier")
    p2 = generate_probe((64, 64), 1.0, 0.05, kind="square", space="fourier")
    assert not np.allclose(np.abs(p1), np.abs(p2))
