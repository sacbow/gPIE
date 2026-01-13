import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.graph.propagator.slice_propagator import SlicePropagator
from gpie.core import AccumulativeUncertainArray

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def run_warm_start(wave, prop, x_msg):
    """
    Run a full-batch warm-start forward pass:
      parent -> wave -> propagator
    """
    wave.parent_message = x_msg
    wave.forward(block=None)
    prop.forward(block=None)


# ----------------------------------------------------------------------
# Initialization tests
# ----------------------------------------------------------------------

def test_slice_propagator_init_invalid_types():
    with pytest.raises(TypeError):
        _ = SlicePropagator("not a slice")

    with pytest.raises(TypeError):
        _ = SlicePropagator((1, 2))

    with pytest.raises(TypeError):
        _ = SlicePropagator([1, 2])

    with pytest.raises(TypeError):
        _ = SlicePropagator([(slice(0, 2), 1)])


def test_slice_propagator_init_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        _ = SlicePropagator([
            (slice(0, 2), slice(0, 2)),
            (slice(0, 3), slice(0, 3)),
        ])


# ----------------------------------------------------------------------
# __matmul__ (graph construction)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_normal_case(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    indices = [(slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4))]

    prop = SlicePropagator(indices)
    out = prop @ wave

    assert out.batch_size == 2
    assert out.event_shape == (2, 2)
    assert out.dtype == wave.dtype
    assert out.parent is prop
    assert prop.output_product is not None
    assert prop.output_product.event_shape == wave.event_shape


@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_batch_size_error(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=2, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 2), slice(0, 2))])

    with pytest.raises(ValueError, match="batch_size=1"):
        _ = prop @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_rank_mismatch(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 2),)])

    with pytest.raises(ValueError, match="rank mismatch"):
        _ = prop @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_out_of_range(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 5), slice(0, 4))])

    with pytest.raises(ValueError, match="out of range"):
        _ = prop @ wave


# ----------------------------------------------------------------------
# Forward / backward (full-batch)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_forward_extracts_patches(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([
        (slice(0, 2), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ])
    _ = prop @ wave

    x_msg = UA.zeros((4, 4), batch_size=1, dtype=xp.complex64, precision=1.0)
    x_msg.data[...] = 1.0

    run_warm_start(wave, prop, x_msg)

    out = prop.last_forward_message
    assert out.batch_size == 2
    assert out.event_shape == (2, 2)
    assert xp.allclose(out.data, 1.0)
    assert xp.allclose(out.precision(raw=False), 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_backward_reconstructs_input(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([
        (slice(0, 2), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ])
    _ = prop @ wave

    x_msg = UA.zeros((4, 4), batch_size=1, dtype=xp.complex64, precision=1.0)
    x_msg.data[...] = 2.0
    run_warm_start(wave, prop, x_msg)

    patch_msg = UA.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    patch_msg.data[...] = 3.0

    prop.receive_message(prop.output, patch_msg, block=None)
    prop.backward(block=None)

    recon = wave.child_messages[prop]
    expected = xp.zeros((4, 4), dtype=xp.complex64)
    expected[0:2, 0:2] = 3.0
    expected[2:4, 2:4] = 3.0

    assert xp.allclose(recon.data[0], expected)


# ----------------------------------------------------------------------
# Forward: block-wise vs full-batch
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_forward_block_matches_full(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 4)),
        (slice(2, 4), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ])
    _ = prop @ wave

    x_msg = UA.zeros((4, 4), batch_size=1, dtype=xp.complex64, precision=1.0)
    x_msg.data[...] = 1.0

    run_warm_start(wave, prop, x_msg)
    full = prop.last_forward_message.copy()

    blocks = [slice(0, 2), slice(2, 4)]
    parts = []

    for blk in blocks:
        wave.forward(block=blk)
        parts.append(prop.last_forward_message.extract_block(blk))

    merged = UA.zeros(
        full.event_shape,
        batch_size=full.batch_size,
        dtype=full.dtype,
        precision=full.precision(raw=True),  # raw precision required
    )
    merged.insert_block(blocks[0], parts[0])
    merged.insert_block(blocks[1], parts[1])

    assert xp.allclose(merged.data, full.data)
    assert xp.allclose(merged.precision(raw=False), full.precision(raw=False))


# ----------------------------------------------------------------------
# Backward: block-wise vs full-batch
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_backward_block_matches_full(xp):
    backend.set_backend(xp)

    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 4)),
        (slice(2, 4), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ])
    _ = prop @ wave

    x_msg = UA.zeros((4, 4), batch_size=1, dtype=xp.complex64, precision=1.0)
    x_msg.data[...] = 1.0
    run_warm_start(wave, prop, x_msg)

    patch_msg = UA.zeros((2, 2), batch_size=4, dtype=xp.complex64, precision=1.0)
    patch_msg.data[...] = 2.0

    # --- full-batch backward (reference)
    prop.receive_message(prop.output, patch_msg, block=None)
    prop.backward(block=None)
    full_back = wave.child_messages[prop].copy()

    # --- reset
    wave.child_messages.clear()
    prop._last_output_update_block = None
    prop._last_output_update_old_block = None

    # --- block-wise backward
    blocks = [slice(0, 2), slice(2, 4)]
    for blk in blocks:
        blk_msg = patch_msg.extract_block(blk)
        prop.receive_message(prop.output, blk_msg, block=blk)
        prop.backward(block=blk)

    block_back = wave.child_messages[prop].copy()

    assert xp.allclose(block_back.data, full_back.data)
    assert xp.allclose(
        block_back.precision(raw=False),
        full_back.precision(raw=False),
    )
