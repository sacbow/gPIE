from typing import List, Tuple


def realspace_to_pixel_coords(
    positions_real: List[Tuple[float, float]],
    pixel_size_um: float,
    obj_shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Convert real-space scan positions (in micrometers) to pixel coordinates.

    The origin in real space (0.0, 0.0) is assumed to map to the center of the object.

    Parameters
    ----------
    positions_real : List of (y_um, x_um)
        Scan positions in real space [μm].
    pixel_size_um : float
        Physical size of 1 pixel [μm/px].
    obj_shape : Tuple[int, int]
        Shape of the object array (H, W) in pixels.

    Returns
    -------
    List of (i, j) positions in pixel coordinates (row, col).
    """
    H, W = obj_shape
    cy, cx = H // 2, W // 2
    coords_pix = []
    for y_um, x_um in positions_real:
        i = int(round(y_um / pixel_size_um + cy))
        j = int(round(x_um / pixel_size_um + cx))
        coords_pix.append((i, j))
    return coords_pix


def filter_positions_within_object(
    pixel_positions: List[Tuple[int, int]],
    obj_shape: Tuple[int, int],
    probe_shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Remove positions that would go out-of-bounds when slicing object with probe.

    Parameters
    ----------
    pixel_positions : list of (i, j)
    obj_shape : (H, W)
    probe_shape : (h, w)

    Returns
    -------
    Filtered list of valid positions.
    """
    H, W = obj_shape
    h, w = probe_shape
    valid = []
    for i, j in pixel_positions:
        if (i - h // 2 >= 0 and i + h // 2 < H and
            j - w // 2 >= 0 and j + w // 2 < W):
            valid.append((i, j))
    return valid



def slices_from_positions(
    pixel_positions: List[Tuple[int, int]],
    probe_shape: Tuple[int, int],
    obj_shape: Tuple[int, int] = None,
) -> List[Tuple[slice, slice]]:
    """
    Convert scan positions in pixel coordinates into slice tuples for object patches.

    Parameters
    ----------
    pixel_positions : List of (i, j)
        Center positions in pixel coordinates (row, col).
    probe_shape : (h, w)
        Shape of the probe window (height, width).
    obj_shape : (H, W), optional
        Shape of the object array. If provided, validity is checked and ValueError raised on overflow.

    Returns
    -------
    List[Tuple[slice, slice]]
        List of (slice_y, slice_x) usable with array indexing.
    """
    h, w = probe_shape
    h_half = h // 2
    w_half = w // 2

    slices = []
    for i, j in pixel_positions:
        y_start = i - h_half
        y_end = y_start + h
        x_start = j - w_half
        x_end = x_start + w

        if obj_shape is not None:
            H, W = obj_shape
            if not (0 <= y_start < y_end <= H and 0 <= x_start < x_end <= W):
                raise ValueError(
                    f"Slice out of bounds: pos=({i},{j}), "
                    f"slice=({y_start}:{y_end}, {x_start}:{x_end}), "
                    f"obj_shape={obj_shape}"
                )

        slices.append((slice(y_start, y_end), slice(x_start, x_end)))

    return slices