import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.multiply_propagator import MultiplyPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.rng_utils import get_rng
from gpie.core.types import (
    PrecisionMode,
    BinaryPropagatorPrecisionMode as BPM,
)

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


def _inject_downstream_message(prop: MultiplyPropagator, z: Wave, ua_z: UA) -> None:
    """
    Inject a 'downstream' message from z to prop.
    This mimics what Graph.backward() would do via downstream factors.
    """
    prop.receive_message(z, ua_z)


# -------------------------------------------------------------------------
# Full-batch warm-start initializes all internal states
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_full_batch_forward_backward_initializes_state(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    prop = MultiplyPropagator()
    a = Wave(event_shape=(2, 2), batch_size=4, dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), batch_size=4, dtype=xp.complex64)
    z = prop @ (a, b)

    ua_a = UA.random((2, 2), batch_size=4, dtype=xp.complex64, rng=rng)
    ua_b = UA.random((2, 2), batch_size=4, dtype=xp.complex64, rng=rng)

    # Inject input messages into propagator
    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b

    # Forward: prop -> z (this sets z.parent_message, and caches prop.last_forward_message)
    prop.forward()

    # Inject a downstream message z -> prop so backward can run
    ua_z_down = UA.random((2, 2), batch_size=4, dtype=xp.complex64, rng=rng)
    _inject_downstream_message(prop, z, ua_z_down)

    # Backward: prop -> a,b
    prop.backward()

    assert prop.output_belief is not None
    assert prop.input_beliefs["a"] is not None
    assert prop.input_beliefs["b"] is not None
    assert prop._last_backward_msgs["a"] is not None
    assert prop._last_backward_msgs["b"] is not None

    # Also ensure waves received messages from prop
    assert isinstance(a.child_messages[prop], UA)
    assert isinstance(b.child_messages[prop], UA)


# -------------------------------------------------------------------------
# Block-wise update affects only the specified block (input beliefs + raw msgs)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_blockwise_backward_updates_only_block(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    prop = MultiplyPropagator()
    a = Wave(event_shape=(2, 2), batch_size=6, dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), batch_size=6, dtype=xp.complex64)
    a._set_precision_mode("array")
    b._set_precision_mode("array")
    z = prop @ (a, b)
    z._set_precision_mode("array")
    ua_a = UA.random((2, 2), batch_size=6, dtype=xp.complex64, rng=rng, scalar_precision = False)
    ua_b = UA.random((2, 2), batch_size=6, dtype=xp.complex64, rng=rng, scalar_precision = False)

    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b

    # Full-batch warm-start: forward + downstream injection + backward
    prop.forward()
    ua_z_down = UA.random((2, 2), batch_size=6, dtype=xp.complex64, rng=rng, scalar_precision = False)
    _inject_downstream_message(prop, z, ua_z_down)
    prop.backward()

    # Snapshot beliefs (data + raw precision) BEFORE block update
    a_data_before = prop.input_beliefs["a"].data.copy()
    a_prec_before = prop.input_beliefs["a"].precision(raw=True).copy()
    b_data_before = prop.input_beliefs["b"].data.copy()
    b_prec_before = prop.input_beliefs["b"].precision(raw=True).copy()

    # Snapshot raw backward messages BEFORE block update
    raw_a_before = prop._last_backward_msgs["a"].data.copy()
    raw_b_before = prop._last_backward_msgs["b"].data.copy()

    blk = slice(2, 4)

    # Block-wise forward/backward
    prop.forward(block=blk)
    # output_message must remain full-batch; inject again (or reuse) to be explicit
    _inject_downstream_message(prop, z, ua_z_down)
    prop.backward(block=blk)

    # --- Block changed ---
    assert not xp.allclose(prop.input_beliefs["a"].data[blk], a_data_before[blk])
    assert not xp.allclose(prop.input_beliefs["b"].data[blk], b_data_before[blk])

    # --- Outside block unchanged (belief means) ---
    assert xp.allclose(prop.input_beliefs["a"].data[:2], a_data_before[:2])
    assert xp.allclose(prop.input_beliefs["a"].data[4:], a_data_before[4:])
    assert xp.allclose(prop.input_beliefs["b"].data[:2], b_data_before[:2])
    assert xp.allclose(prop.input_beliefs["b"].data[4:], b_data_before[4:])

    # --- Outside block unchanged (raw VMP messages) ---
    assert xp.allclose(prop._last_backward_msgs["a"].data[:2], raw_a_before[:2])
    assert xp.allclose(prop._last_backward_msgs["a"].data[4:], raw_a_before[4:])
    assert xp.allclose(prop._last_backward_msgs["b"].data[:2], raw_b_before[:2])
    assert xp.allclose(prop._last_backward_msgs["b"].data[4:], raw_b_before[4:])

    # Precision outside block should also be unchanged (raw precision)
    assert xp.allclose(prop.input_beliefs["a"].precision(raw=True)[:2], a_prec_before[:2])
    assert xp.allclose(prop.input_beliefs["a"].precision(raw=True)[4:], a_prec_before[4:])
    assert xp.allclose(prop.input_beliefs["b"].precision(raw=True)[:2], b_prec_before[:2])
    assert xp.allclose(prop.input_beliefs["b"].precision(raw=True)[4:], b_prec_before[4:])


# -------------------------------------------------------------------------
# Scalar precision backward projection
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_backward_scalar_precision_projection(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    prop = MultiplyPropagator(precision_mode=BPM.SCALAR_AND_ARRAY_TO_ARRAY)
    a = Wave(event_shape=(2, 2), batch_size=3, dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), batch_size=3, dtype=xp.complex64)
    z = prop @ (a, b)

    # Force precision modes
    a._set_precision_mode("scalar")
    b._set_precision_mode("array")

    ua_a = UA.random(
        (2, 2),
        batch_size=3,
        dtype=xp.complex64,
        rng=rng,
        scalar_precision=True,
    )
    ua_b = UA.random(
        (2, 2),
        batch_size=3,
        dtype=xp.complex64,
        rng=rng,
        scalar_precision=False,
    )

    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b

    prop.forward()
    ua_z_down = UA.random((2, 2), batch_size=3, dtype=xp.complex64, rng=rng, scalar_precision=False)
    _inject_downstream_message(prop, z, ua_z_down)
    prop.backward()

    msg_to_a = a.child_messages[prop]
    assert isinstance(msg_to_a, UA)
    assert msg_to_a._scalar_precision


# -------------------------------------------------------------------------
# Precision mode propagation (logic only)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_precision_mode_propagation_forward_scalar_scalar_raises(xp):
    backend.set_backend(xp)

    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)

    a._set_precision_mode("scalar")
    b._set_precision_mode("scalar")

    with pytest.raises(ValueError):
        z.parent.set_precision_mode_forward()


# -------------------------------------------------------------------------
# Repr & sample generation
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_repr_and_sample_generation(xp):
    backend.set_backend(xp)

    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    prop = z.parent

    a.set_sample(xp.ones((2, 2)))
    b.set_sample(2 * xp.ones((2, 2)))

    sample = prop.get_sample_for_output(get_rng(0))
    assert xp.allclose(sample, 2.0)

    rep = repr(prop)
    assert "Mul" in rep and "mode" in rep


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_block_without_full_init_raises(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    prop = MultiplyPropagator()
    a = Wave(event_shape=(2, 2), batch_size=4, dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), batch_size=4, dtype=xp.complex64)
    z = prop @ (a, b)

    ua_a = UA.random((2, 2), batch_size=4, dtype=xp.complex64, rng=rng)
    ua_b = UA.random((2, 2), batch_size=4, dtype=xp.complex64, rng=rng)

    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b

    # No full-batch forward yet
    with pytest.raises(RuntimeError):
        prop.forward(block=slice(0, 2))

@pytest.mark.parametrize("xp", backend_libs)
def test_backward_without_output_message_raises(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    prop = MultiplyPropagator()
    a = Wave(event_shape=(2, 2), batch_size=3, dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), batch_size=3, dtype=xp.complex64)
    z = prop @ (a, b)

    ua_a = UA.random((2, 2), batch_size=3, dtype=xp.complex64, rng=rng)
    ua_b = UA.random((2, 2), batch_size=3, dtype=xp.complex64, rng=rng)

    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b

    prop.forward()
    # no downstream injection

    with pytest.raises(RuntimeError):
        prop.backward()

@pytest.mark.parametrize("xp", backend_libs)
def test_blockwise_backward_without_full_init_raises(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    prop = MultiplyPropagator()
    a = Wave(event_shape=(2, 2), batch_size=5, dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), batch_size=5, dtype=xp.complex64)
    z = prop @ (a, b)

    ua_a = UA.random((2, 2), batch_size=5, dtype=xp.complex64, rng=rng)
    ua_b = UA.random((2, 2), batch_size=5, dtype=xp.complex64, rng=rng)

    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b

    # full forward only
    prop.forward()
    ua_z_down = UA.random((2, 2), batch_size=5, dtype=xp.complex64, rng=rng)
    _inject_downstream_message(prop, z, ua_z_down)

    # no full backward yet → raw buffers missing
    with pytest.raises(RuntimeError):
        prop.backward(block=slice(1, 3))
