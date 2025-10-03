import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import pytest
from examples.ptychography.simulator.scan import generate_raster_positions
import math

#-------- raster scan ---------

def test_generate_raster_positions_output_format():
    gen = generate_raster_positions(stride_um=1.0)
    for _ in range(10):
        pos = next(gen)
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert all(isinstance(coord, float) for coord in pos)


def test_generate_raster_positions_ordering():
    """
    Test that spiral pattern expands outward from origin and radius increases.
    """
    gen = generate_raster_positions(stride_um=1.0)
    coords = [next(gen) for _ in range(25)]

    # Compute distance squared from origin
    dists = [y**2 + x**2 for y, x in coords]

    # Check that distance increases *on average*
    # (Not strictly monotonic due to spiral, but trend should rise)
    assert dists[0] == 0.0
    assert dists[-1] > dists[0]
    assert dists[10] < dists[20]


def test_generate_raster_positions_stride_scaling():
    """
    Test that stride_um correctly scales the output coordinates.
    """
    gen1 = generate_raster_positions(stride_um=1.0)
    gen2 = generate_raster_positions(stride_um=2.0)

    for _ in range(10):
        p1 = next(gen1)
        p2 = next(gen2)
        assert math.isclose(p2[0], p1[0] * 2.0, rel_tol=1e-6)
        assert math.isclose(p2[1], p1[1] * 2.0, rel_tol=1e-6)


def test_generate_many_positions_no_crash():
    """
    Make sure generator can run for many steps without error.
    """
    gen = generate_raster_positions(stride_um=1.0)
    for _ in range(1000):
        pos = next(gen)
        assert isinstance(pos, tuple)

#-------- fermat spiral ---------

from examples.ptychography.simulator.scan import generate_fermat_spiral_positions


def test_generate_fermat_spiral_output_format():
    gen = generate_fermat_spiral_positions(step_um=1.0)
    for _ in range(10):
        pos = next(gen)
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert all(isinstance(coord, float) for coord in pos)


def test_generate_fermat_spiral_distance_increases():
    gen = generate_fermat_spiral_positions(step_um=1.0)
    coords = [next(gen) for _ in range(50)]
    dists = [y**2 + x**2 for y, x in coords]

    # Fermat spiral: distance increases over time
    assert dists[0] == 0.0
    assert dists[-1] > dists[0]
    assert dists[10] < dists[20] < dists[30] < dists[40]  # general trend


def test_generate_fermat_spiral_step_scaling():
    gen1 = generate_fermat_spiral_positions(step_um=1.0)
    gen2 = generate_fermat_spiral_positions(step_um=2.0)

    for _ in range(20):
        p1 = next(gen1)
        p2 = next(gen2)
        dist1 = p1[0] ** 2 + p1[1] ** 2
        dist2 = p2[0] ** 2 + p2[1] ** 2

        # Ratio of radii squared ≈ 4 if step is doubled
        if dist1 > 0:
            ratio = dist2 / dist1
            assert 3.5 < ratio < 4.5  # Allow some tolerance
