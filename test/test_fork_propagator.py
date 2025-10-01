import pytest
import numpy as np

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.graph.wave import Wave
from gpie.graph.propagator.fork_propagator import ForkPropagator
from gpie.graph.prior.gaussian_prior import GaussianPrior
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode


def test_forward_then_backward_cycle():
    backend.set_backend(np)
    # Prior → Wave
    wave_in = ~GaussianPrior(event_shape=(2, 2), dtype=np.complex64)
    prior = wave_in.parent
    msg_from_prior = UA.random(event_shape=(2, 2), batch_size=1, precision=1.0)
    wave_in.receive_message(prior, msg_from_prior)

    # Fork
    fork = ForkPropagator(batch_size=2, dtype=wave_in.dtype)
    wave_out = fork @ wave_in

    # Forward pass
    wave_in.forward()
    fork.forward()
    assert wave_out.parent_message is not None
    assert wave_out.parent_message.batch_size == 2

    # Backward pass
    msg_from_output = UA.random(event_shape=(2, 2), batch_size=2, precision=1.0)
    fork.receive_message(wave_out, msg_from_output)
    fork.backward()
    assert wave_in.child_messages[fork] is not None
    assert wave_in.child_messages[fork].batch_size == 1


def test_incremental_update_changes_message():
    backend.set_backend(np)

    ua1 = UA.random(event_shape=(3,), batch_size=1, precision=1.0, scalar_precision=True)
    ua2 = UA.random(event_shape=(3,), batch_size=1, precision=2.0, scalar_precision=True)

    wave_in = Wave(event_shape=(3,), batch_size=1, dtype=ua1.dtype)
    fork = ForkPropagator(batch_size=2, dtype=ua1.dtype)
    _ = fork @ wave_in

    # First forward
    msg1 = fork._compute_forward({"input": ua1})
    assert msg1.batch_size == 2
    fork.output_message = msg1

    # Second forward with updated input
    msg2 = fork._compute_forward({"input": ua2})
    assert msg2.batch_size == 2
    # 内容が変わっているはず
    assert not np.allclose(msg1.data, msg2.data)


def test_invalid_batch_size_and_input_wave():
    with pytest.raises(ValueError):
        _ = ForkPropagator(batch_size=0)

    backend.set_backend(np)
    wave = Wave(event_shape=(2, 2), batch_size=2, dtype=np.complex64)
    fork = ForkPropagator(batch_size=3)
    with pytest.raises(ValueError):
        _ = fork @ wave


def test_precision_mode_setting_and_propagation():
    backend.set_backend(np)

    # --- SCALAR mode ---
    fork = ForkPropagator(batch_size=2)
    fork._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR)
    assert fork.get_output_precision_mode() == PrecisionMode.SCALAR
    assert fork.get_input_precision_mode(None) == PrecisionMode.SCALAR

    # --- ARRAY mode ---
    fork = ForkPropagator(batch_size=2)
    fork._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
    assert fork.get_output_precision_mode() == PrecisionMode.ARRAY
    assert fork.get_input_precision_mode(None) == PrecisionMode.ARRAY

    # --- Forward propagation sets mode ---
    wave_in = Wave(event_shape=(2,), batch_size=1, dtype=np.complex64, precision_mode=PrecisionMode.SCALAR)
    fork = ForkPropagator(batch_size=2)
    _ = fork @ wave_in
    fork.set_precision_mode_forward()
    assert fork.precision_mode_enum in (
        UnaryPropagatorPrecisionMode.SCALAR,
        UnaryPropagatorPrecisionMode.ARRAY,
    )

    # --- Backward propagation sets mode ---
    fork.set_precision_mode_backward()
    assert fork.precision_mode_enum in (
        UnaryPropagatorPrecisionMode.SCALAR,
        UnaryPropagatorPrecisionMode.ARRAY,
    )


def test_get_input_output_precision_mode_none():
    fork = ForkPropagator(batch_size=2)
    assert fork.get_input_precision_mode(None) is None
    assert fork.get_output_precision_mode() is None
