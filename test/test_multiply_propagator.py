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


# -------------------------------------------------------------------------
# Updated test: verify variational inference (replaces _compute_forward)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_compute_variational_inference_array_to_array(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    prop = MultiplyPropagator()
    a = Wave(event_shape=(2, 2), dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), dtype=xp.complex64)
    z = prop @ (a, b)
    prop.set_init_rng(rng)

    ua_a = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)
    ua_b = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)
    ua_z = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)

    prop.input_beliefs["a"] = ua_a
    prop.input_beliefs["b"] = ua_b
    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b
    prop.output_message = ua_z

    prop.compute_variational_inference()

    assert isinstance(prop.output_belief, UA)
    assert prop.output_belief.event_shape == (2, 2)
    assert not prop.output_belief._scalar_precision
    assert isinstance(prop.input_beliefs["a"], UA)
    assert isinstance(prop.input_beliefs["b"], UA)


# -------------------------------------------------------------------------
# Forward random initialization (unchanged)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_forward_random_initialization(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    a = Wave(event_shape=(2, 2), dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), dtype=xp.complex64)
    z = MultiplyPropagator() @ (a, b)
    prop = z.parent
    prop.set_init_rng(rng)

    prop.forward()
    assert isinstance(z.parent_message, UA)


# -------------------------------------------------------------------------
# Backward updates input beliefs (main backward check)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_backward_updates_input_beliefs(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    a = Wave(event_shape=(2, 2), dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), dtype=xp.complex64)
    z = MultiplyPropagator() @ (a, b)
    prop = z.parent
    prop.set_init_rng(rng)

    ua_a = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)
    ua_b = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)
    ua_z = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)

    prop.input_beliefs["a"] = ua_a
    prop.input_beliefs["b"] = ua_b
    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b
    prop.receive_message(z, ua_z)

    prop.backward()

    assert isinstance(prop.input_beliefs["a"], UA)
    assert isinstance(prop.input_beliefs["b"], UA)
    assert isinstance(a.child_messages[prop], UA)
    assert isinstance(b.child_messages[prop], UA)


# -------------------------------------------------------------------------
# Backward scalar projection (replaces _compute_backward test)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_backward_scalar_projection(xp):
    backend.set_backend(xp)
    rng = get_rng(0)

    a = Wave(event_shape=(2, 2), dtype=xp.complex64)
    b = Wave(event_shape=(2, 2), dtype=xp.complex64)
    z = MultiplyPropagator(precision_mode=BPM.SCALAR_AND_ARRAY_TO_ARRAY) @ (a, b)
    prop = z.parent
    prop.set_init_rng(rng)

    a._set_precision_mode("scalar")
    b._set_precision_mode("array")

    ua_a = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=True)
    ua_b = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)
    ua_z = UA.random(event_shape=(2, 2), dtype=xp.complex64, rng=rng, scalar_precision=False)

    prop.input_messages[a] = ua_a
    prop.input_messages[b] = ua_b
    prop.input_beliefs["a"] = ua_a
    prop.input_beliefs["b"] = ua_b
    prop.receive_message(z, ua_z)

    prop.backward()

    msg_to_a = a.child_messages[prop]
    assert isinstance(msg_to_a, UA)
    assert msg_to_a._scalar_precision


# -------------------------------------------------------------------------
# Precision mode propagation (unchanged)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_precision_mode_propagation_forward_and_backward(xp):
    backend.set_backend(xp)

    # --- forward: SCALAR × SCALAR → ValueError ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    a._set_precision_mode("scalar")
    b._set_precision_mode("scalar")
    with pytest.raises(ValueError):
        z.parent.set_precision_mode_forward()

    # --- forward: SCALAR × ARRAY → SCALAR_AND_ARRAY_TO_ARRAY ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    a._set_precision_mode("scalar")
    b._set_precision_mode("array")
    z.parent.set_precision_mode_forward()
    assert z.parent.precision_mode_enum == BPM.SCALAR_AND_ARRAY_TO_ARRAY

    # --- forward: ARRAY × SCALAR → ARRAY_AND_SCALAR_TO_ARRAY ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    a._set_precision_mode("array")
    b._set_precision_mode("scalar")
    z.parent.set_precision_mode_forward()
    assert z.parent.precision_mode_enum == BPM.ARRAY_AND_SCALAR_TO_ARRAY

    # --- forward: ARRAY × ARRAY → ARRAY ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    a._set_precision_mode("array")
    b._set_precision_mode("array")
    z.parent.set_precision_mode_forward()
    # forwardではARRAY指定で完了する場合もあるが、明示的にset_precision_mode_backward()で確定
    z._set_precision_mode("array")
    z.parent.set_precision_mode_backward()
    assert z.parent.precision_mode_enum == BPM.ARRAY

    # --- backward: z.scalar + a,b unset → ARRAY_AND_ARRAY_TO_SCALAR ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    z._set_precision_mode("scalar")
    z.parent.set_precision_mode_backward()
    assert z.parent.precision_mode_enum == BPM.ARRAY_AND_ARRAY_TO_SCALAR

    # --- backward: z.array + a.scalar, b.unset → SCALAR_AND_ARRAY_TO_ARRAY ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    z._set_precision_mode("array")
    a._set_precision_mode("scalar")
    z.parent.set_precision_mode_backward()
    assert z.parent.precision_mode_enum == BPM.SCALAR_AND_ARRAY_TO_ARRAY

    # --- backward: z.array + a.unset, b.scalar → ARRAY_AND_SCALAR_TO_ARRAY ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    z._set_precision_mode("array")
    b._set_precision_mode("scalar")
    z.parent.set_precision_mode_backward()
    assert z.parent.precision_mode_enum == BPM.ARRAY_AND_SCALAR_TO_ARRAY

    # --- backward: z.array + a,b unset → ARRAY ---
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    z = MultiplyPropagator() @ (a, b)
    z._set_precision_mode("array")
    z.parent.set_precision_mode_backward()
    assert z.parent.precision_mode_enum == BPM.ARRAY


# -------------------------------------------------------------------------
# Repr & sample test (unchanged)
# -------------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_repr_and_generate_sample(xp):
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
