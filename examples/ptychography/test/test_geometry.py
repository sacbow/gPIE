import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import pytest
from examples.ptychography.utils.geometry import realspace_to_pixel_coords, filter_positions_within_object


def test_realspace_to_pixel_coords_center_mapping():
    """
    (0,0) in real space should map to the center pixel of the object.
    """
    obj_shape = (512, 512)
    pixel_size_um = 0.1
    real_positions = [(0.0, 0.0)]
    pixel_positions = realspace_to_pixel_coords(real_positions, pixel_size_um, obj_shape)

    H, W = obj_shape
    cy, cx = H // 2, W // 2
    assert pixel_positions[0] == (cy, cx)


def test_realspace_to_pixel_coords_scaling():
    """
    Positions in real space should be scaled by pixel_size_um correctly.
    """
    obj_shape = (512, 512)
    pixel_size_um = 0.1  # μm per pixel
    real_positions = [(1.0, 1.0), (-1.0, -1.0)]
    pixel_positions = realspace_to_pixel_coords(real_positions, pixel_size_um, obj_shape)

    cy, cx = obj_shape[0] // 2, obj_shape[1] // 2
    # 1.0 μm offset = 1.0/0.1 = 10 pixel offset
    assert pixel_positions[0] == (cy + 10, cx + 10)
    assert pixel_positions[1] == (cy - 10, cx - 10)


def test_realspace_to_pixel_coords_multiple_positions():
    """
    Should handle multiple positions and return same length list.
    """
    obj_shape = (256, 256)
    pixel_size_um = 0.5
    real_positions = [(0.0, 0.0), (5.0, -5.0), (-2.5, 2.5)]
    pixel_positions = realspace_to_pixel_coords(real_positions, pixel_size_um, obj_shape)

    assert len(pixel_positions) == len(real_positions)
    # First one should be center
    cy, cx = obj_shape[0] // 2, obj_shape[1] // 2
    assert pixel_positions[0] == (cy, cx)

def test_filter_positions_within_object_basic():
    obj_shape = (100, 100)
    probe_shape = (20, 20)

    # 中心付近（全部有効）
    positions = [(50, 50), (40, 40), (60, 60)]
    filtered = filter_positions_within_object(positions, obj_shape, probe_shape)
    assert filtered == positions


def test_filter_positions_within_object_out_of_bounds():
    obj_shape = (100, 100)
    probe_shape = (20, 20)

    positions = [
        (10, 10),  # valid
        (0, 0),    # invalid
        (95, 95),  # invalid
        (50, 90),  # partially out
        (50, 50)   # valid
    ]
    filtered = filter_positions_within_object(positions, obj_shape, probe_shape)
    assert (0, 0) not in filtered
    assert (95, 95) not in filtered
    assert (50, 90) not in filtered
    assert (10, 10) in filtered
    assert (50, 50) in filtered
    assert len(filtered) == 2


def test_filter_positions_empty_when_all_outside():
    obj_shape = (64, 64)
    probe_shape = (32, 32)

    positions = [(0, 0), (63, 63), (5, 60), (60, 5)]
    filtered = filter_positions_within_object(positions, obj_shape, probe_shape)
    assert filtered == []