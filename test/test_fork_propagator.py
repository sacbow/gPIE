# test/test_fork_propagator.py

import pytest
import numpy as np

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.graph.wave import Wave
from gpie.graph.propagator.fork_propagator import ForkPropagator
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode


def _make_scalar_ua(event_shape, batch_size=1, dtype=np.complex64, precision=1.0, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(batch_size, *event_shape)) + 1j * rng.normal(size=(batch_size, *event_shape))
    data = data.astype(dtype)
    # scalar precision -> use UA.random for consistency (but here deterministic for tests)
    # UA constructor expects precision broadcast rules handled inside; simplest: use UA.random in real tests
    ua = UA(data, dtype=dtype, precision=np.array(precision, dtype=np.float32), batched=(batch_size != 1))
    # Ensure scalar precision explicitly
    return ua.as_scalar_precision()


def _send_input_message_to_fork(wave_in: Wave, fork: ForkPropagator, ua_in: UA):
    """
    Minimal contract helper:
      - deliver message to wave_in
      - run wave_in.forward() so fork receives message (propagator.input_messages populated)
    """
    wave_in.receive_message(None, ua_in)
    wave_in.forward()  # pushes message to child propagators (including fork)


def test_fork_forward_backward_basic_contract():
    backend.set_backend(np)

    event_shape = (2, 2)
    B_out = 3

    # input wave (batch_size=1)
    wave_in = Wave(event_shape=event_shape, batch_size=1, dtype=np.complex64)

    # fork propagator
    fork = ForkPropagator(batch_size=B_out, dtype=np.complex64)
    wave_out = fork @ wave_in
    fork._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)

    # provide input message and push it to fork
    ua_in = UA.random(event_shape=event_shape, batch_size=1, dtype=np.complex64, precision=1.0)
    _send_input_message_to_fork(wave_in, fork, ua_in)

    # forward: fork -> output wave
    fork.forward()
    out_msg = wave_out.parent_message
    assert out_msg is not None
    assert out_msg.batch_size == B_out
    assert out_msg.event_shape == event_shape

    # provide output-side message and backward: fork -> input wave
    ua_out = UA.random(event_shape=event_shape, batch_size=B_out, dtype=np.complex64, precision=1.0)
    fork.receive_message(wave_out, ua_out)
    fork.backward()

    in_msg = wave_in.child_messages[fork]
    assert in_msg is not None
    assert in_msg.batch_size == 1
    assert in_msg.event_shape == event_shape


def test_fork_forward_initial_no_output_message_returns_forked_belief():
    backend.set_backend(np)

    event_shape = (2, 2)
    B_out = 4
    wave_in = Wave(event_shape=event_shape, batch_size=1, dtype=np.complex64)
    fork = ForkPropagator(batch_size=B_out, dtype=np.complex64)
    fork._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
    wave_out = fork @ wave_in

    ua_in = UA.random(event_shape=event_shape, batch_size=1, dtype=np.complex64, precision=1.0)
    _send_input_message_to_fork(wave_in, fork, ua_in)

    # No output_message has been received yet
    assert fork.output_message is None

    fork.forward()
    out_msg = wave_out.parent_message
    assert out_msg is not None
    assert out_msg.batch_size == B_out

    # In the first iteration, it should be a simple fork() of belief (= input, since child_product is None)
    expected = ua_in.fork(batch_size=B_out)
    assert np.allclose(out_msg.data, expected.data)
    assert np.allclose(out_msg.precision(raw=True), expected.precision(raw=True))


def test_fork_backward_full_batch_equivalence_to_product_reduce():
    backend.set_backend(np)

    event_shape = (3,)
    B_out = 5
    wave_in = Wave(event_shape=event_shape, batch_size=1, dtype=np.complex64)
    fork = ForkPropagator(batch_size=B_out, dtype=np.complex64)
    fork._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
    wave_out = fork @ wave_in

    ua_in = UA.random(event_shape=event_shape, batch_size=1, dtype=np.complex64, precision=1.0)
    _send_input_message_to_fork(wave_in, fork, ua_in)

    # produce a forward message so caches are initialized (block_size = full batch)
    fork.forward()

    # create output message and run backward
    ua_out = UA.random(event_shape=event_shape, batch_size=B_out, dtype=np.complex64, precision=1.0)
    fork.receive_message(wave_out, ua_out)

    fork.backward(block=None)

    in_msg = wave_in.child_messages[fork]
    assert in_msg is not None
    assert in_msg.batch_size == 1

    expected = ua_out.product_reduce_over_batch()
    assert np.allclose(in_msg.data, expected.data)
    assert np.allclose(in_msg.precision(raw=True), expected.precision(raw=True))


def test_invalid_batch_size_and_input_wave():
    backend.set_backend(np)

    with pytest.raises(ValueError):
        _ = ForkPropagator(batch_size=0)

    wave_bad = Wave(event_shape=(2, 2), batch_size=2, dtype=np.complex64)
    fork = ForkPropagator(batch_size=3, dtype=np.complex64)
    with pytest.raises(ValueError):
        _ = fork @ wave_bad


def test_precision_mode_setting_and_getters():
    backend.set_backend(np)

    fork = ForkPropagator(batch_size=2)

    # direct setter: only SCALAR/ARRAY allowed
    fork._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
    assert fork.get_input_precision_mode(None) == PrecisionMode.SCALAR
    assert fork.get_output_precision_mode() == PrecisionMode.SCALAR

    fork = ForkPropagator(batch_size=2)
    fork._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
    assert fork.get_input_precision_mode(None) == PrecisionMode.ARRAY
    assert fork.get_output_precision_mode() == PrecisionMode.ARRAY

    # invalid mode rejected
    fork = ForkPropagator(batch_size=2)
    with pytest.raises(ValueError):
        fork._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)


def test_precision_mode_propagation_forward_backward():
    backend.set_backend(np)

    # forward propagation uses input wave precision
    wave_in_1 = Wave(event_shape=(2,), batch_size=1, dtype=np.complex64)
    fork_1 = ForkPropagator(batch_size=2)
    _ = fork_1 @ wave_in_1

    wave_in_1._set_precision_mode(PrecisionMode.SCALAR)
    fork_1.set_precision_mode_forward()
    assert fork_1.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR

    wave_in_2 = Wave(event_shape=(2,), batch_size=1, dtype=np.complex64)
    wave_in_2._set_precision_mode(PrecisionMode.ARRAY)
    fork_2 = ForkPropagator(batch_size=2)
    _ = fork_2 @ wave_in_2
    fork_2.set_precision_mode_forward()
    assert fork_2.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY

    # backward propagation uses output wave precision
    wave_in_3 = Wave(event_shape=(2,), batch_size=1, dtype=np.complex64)
    fork_3 = ForkPropagator(batch_size=2)
    wave_out_3 = fork_3 @ wave_in_3
    wave_out_3._set_precision_mode(PrecisionMode.SCALAR)
    fork_3.set_precision_mode_backward()
    assert fork_3.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR

    wave_in_4 = Wave(event_shape=(2,), batch_size=1, dtype=np.complex64)
    fork_4 = ForkPropagator(batch_size=2)
    wave_out_4 = fork_4 @ wave_in_4
    wave_out_4._set_precision_mode(PrecisionMode.ARRAY)
    fork_4.set_precision_mode_backward()
    assert fork_4.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY


def test_get_input_output_precision_mode_none():
    fork = ForkPropagator(batch_size=2)
    assert fork.get_input_precision_mode(None) is None
    assert fork.get_output_precision_mode() is None
