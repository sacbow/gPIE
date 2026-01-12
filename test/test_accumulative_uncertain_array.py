import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.core.accumulative_uncertain_array import AccumulativeUncertainArray
from gpie.core.uncertain_array import UncertainArray

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_repr_and_clear(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [(slice(0, 2), slice(0, 2))]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with one patch: fill with ones, precision=1
    ua = UncertainArray.zeros((2, 2), batch_size=1, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    assert xp.any(aua.precision != 0)
    assert xp.any(aua.weighted_data != 0)

    # clear must reset both arrays
    aua.clear()
    assert xp.allclose(aua.precision, 0)
    assert xp.allclose(aua.weighted_data, 0)

    # repr sanity check
    rep = repr(aua)
    assert "AUA" in rep and "event_shape" in rep


@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_mul_and_extract_patches(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with two patches: all zeros, precision=1
    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    aua.scatter_mul(ua)

    # extract_patches should return same UA back
    ua_out = aua.extract_patches()
    assert ua_out.batch_size == 2
    assert ua_out.event_shape == (2, 2)
    assert xp.allclose(ua_out.data, xp.zeros_like(ua_out.data))
    assert xp.allclose(ua_out.precision(raw=False), xp.ones_like(ua_out.data))


@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_mul_with_overlapping_indices(xp):
    backend.set_backend(xp)
    event_shape = (3, 3)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(1, 3), slice(1, 3)),  # overlaps with first
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with two patches: all ones, precision=1
    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    # Overlap at (1,1) should accumulate precision=2, weighted=2
    total_prec = aua.precision[1, 1]
    total_weighted = aua.weighted_data[1, 1]
    assert xp.allclose(total_prec, 2.0)
    assert xp.allclose(total_weighted, 2.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_as_uncertain_array(xp):
    backend.set_backend(xp)
    event_shape = (2, 2)
    indices = [(slice(0, 2), slice(0, 2))]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with one patch: ones, precision=2
    ua = UncertainArray.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=2.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    ua_out = aua.as_uncertain_array()
    assert ua_out.batch_size == 1
    assert ua_out.event_shape == (2, 2)
    assert xp.allclose(ua_out.data, xp.ones_like(ua_out.data))
    assert xp.allclose(ua_out.precision(raw=False), xp.full((1, 2, 2), 2.0))


@pytest.mark.parametrize("xp", backend_libs)
def test_mul_ua(xp):
    backend.set_backend(xp)
    event_shape = (3, 3)
    indices = [(slice(0, 3), slice(0, 3))]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with batch_size=1, event_shape=(3,3), precision=2, data=1
    ua = UncertainArray.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=2.0)
    ua.data[...] = 1.0

    aua.mul_ua(ua)

    # precision must be filled with 2.0
    assert xp.allclose(aua.precision, xp.full(event_shape, 2.0))
    # weighted_data must be data * precision = 1*2 = 2
    assert xp.allclose(aua.weighted_data, xp.full(event_shape, 2.0))

    # apply mul_ua again → should accumulate
    aua.mul_ua(ua)
    assert xp.allclose(aua.precision, xp.full(event_shape, 4.0))
    assert xp.allclose(aua.weighted_data, xp.full(event_shape, 4.0))


@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_add_blockwise_matches_full_batch(xp):
    backend.set_backend(xp)

    event_shape = (4, 4)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 4)),
        (slice(2, 4), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]

    # Prepare UA: 4 patches, data=1, precision=1
    ua = UncertainArray.zeros(
        (2, 2),
        batch_size=4,
        dtype=xp.complex64,
        precision=1.0,
    )
    ua.data[...] = 1.0

    # --- Full-batch path ---
    aua_full = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)
    aua_full.scatter_mul(ua)

    # --- Block-wise path ---
    aua_block = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # simulate block-wise update (block_size=2)
    blocks = [slice(0, 2), slice(2, 4)]
    for blk in blocks:
        aua_block.scatter_add_ua(ua, block=blk)

    # Compare results
    assert xp.allclose(aua_block.precision, aua_full.precision)
    assert xp.allclose(aua_block.weighted_data, aua_full.weighted_data)


@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_add_then_sub_restores_state(xp):
    backend.set_backend(xp)

    event_shape = (3, 3)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(1, 3), slice(1, 3)),  # overlap
    ]

    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # Initial UA: two patches, data=1, precision=1
    ua = UncertainArray.zeros(
        (2, 2),
        batch_size=2,
        dtype=xp.complex64,
        precision=1.0,
    )
    ua.data[...] = 1.0

    # Save initial state
    prec0 = aua.precision.copy()
    wdata0 = aua.weighted_data.copy()

    # Apply add on block 0
    blk = slice(0, 1)
    aua.scatter_add_ua(ua, block=blk)

    # Apply subtract on same block
    aua.scatter_sub_ua(ua, block=blk)

    # Must be restored exactly
    assert xp.allclose(aua.precision, prec0)
    assert xp.allclose(aua.weighted_data, wdata0)


@pytest.mark.parametrize("xp", backend_libs)
def test_multiple_blocks_add_and_sub_sequence(xp):
    backend.set_backend(xp)

    event_shape = (4, 4)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 4)),
        (slice(2, 4), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]

    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    ua = UncertainArray.zeros(
        (2, 2),
        batch_size=4,
        dtype=xp.complex64,
        precision=1.0,
    )
    ua.data[...] = 1.0

    # Full scatter for reference
    aua_ref = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)
    aua_ref.scatter_mul(ua)

    # Block-wise add
    blocks = [slice(0, 2), slice(2, 4)]
    for blk in blocks:
        aua.scatter_add_ua(ua, block=blk)

    # Block-wise subtract in reverse order
    for blk in reversed(blocks):
        aua.scatter_sub_ua(ua, block=blk)

    # Must be back to zero
    assert xp.allclose(aua.precision, 0)
    assert xp.allclose(aua.weighted_data, 0)

    # Re-apply adds → must match reference again
    for blk in blocks:
        aua.scatter_add_ua(ua, block=blk)

    assert xp.allclose(aua.precision, aua_ref.precision)
    assert xp.allclose(aua.weighted_data, aua_ref.weighted_data)

@pytest.mark.parametrize("xp", backend_libs)
def test_extract_patches_block_none_matches_full(xp):
    backend.set_backend(xp)

    event_shape = (4, 4)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    ua_full = aua.extract_patches()
    ua_block_none = aua.extract_patches(block=None)

    assert xp.allclose(ua_full.data, ua_block_none.data)
    assert xp.allclose(
        ua_full.precision(raw=False),
        ua_block_none.precision(raw=False),
    )

@pytest.mark.parametrize("xp", backend_libs)
def test_extract_patches_block_subset(xp):
    backend.set_backend(xp)

    event_shape = (4, 4)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 4)),
        (slice(2, 4), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    ua = UncertainArray.zeros((2, 2), batch_size=4, dtype=xp.complex64, precision=1.0)
    ua.data[...] = xp.arange(4).reshape(4, 1, 1)
    aua.scatter_mul(ua)

    blk = slice(1, 3)
    ua_blk = aua.extract_patches(block=blk)

    assert ua_blk.batch_size == 2

    # Expected values from full extract
    ua_full = aua.extract_patches()
    assert xp.allclose(
        ua_blk.data,
        ua_full.data[blk],
    )
    assert xp.allclose(
        ua_blk.precision(raw=False),
        ua_full.precision(raw=False)[blk],
    )


@pytest.mark.parametrize("xp", backend_libs)
def test_extract_patches_block_with_overlap(xp):
    backend.set_backend(xp)

    event_shape = (3, 3)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(1, 3), slice(1, 3)),  # overlap
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    ua_full = aua.extract_patches()

    blk = slice(1, 2)
    ua_blk = aua.extract_patches(block=blk)

    assert ua_blk.batch_size == 1
    assert xp.allclose(ua_blk.data, ua_full.data[blk])
    assert xp.allclose(
        ua_blk.precision(raw=False),
        ua_full.precision(raw=False)[blk],
    )
