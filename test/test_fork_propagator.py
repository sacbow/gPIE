import pytest
import numpy as np

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.graph.wave import Wave
from gpie.graph.propagator.fork_propagator import ForkPropagator
from gpie.graph.prior import GaussianPrior
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode


def test_forward_then_backward_updates_belief():
    backend.set_backend(np)
    wave_in = ~GaussianPrior(event_shape=(2, 2), dtype=np.complex64)
    prior = wave_in.parent
    msg_from_prior = UA.random(event_shape=(2, 2), batch_size=1, precision=1.0)
    wave_in.receive_message(prior, msg_from_prior)

    fork = ForkPropagator(batch_size=2, dtype=wave_in.dtype)
    wave_out = fork @ wave_in

    wave_in.forward()
    fork.forward()
    assert wave_out.parent_message.batch_size == 2
    assert fork.belief is not None

    msg_from_output = UA.random(event_shape=(2, 2), batch_size=2, precision=1.0)
    fork.receive_message(wave_out, msg_from_output)
    fork.backward()
    assert wave_in.child_messages[fork].batch_size == 1
    assert fork.belief is not None


def test_incremental_update_changes_belief_and_preserves_precision():
    backend.set_backend(np)

    ua1 = UA.random(event_shape=(3,), batch_size=1, precision=1.0, scalar_precision=True)
    ua2 = UA.random(event_shape=(3,), batch_size=1, precision=2.0, scalar_precision=True)

    wave_in = Wave(event_shape=(3,), batch_size=1, dtype=ua1.dtype)
    wave_in.receive_message(None, ua1)

    fork = ForkPropagator(batch_size=2, dtype=ua1.dtype)
    _ = fork @ wave_in

    # 1回目 forward: belief/output_message 初期化
    msg1 = fork._compute_forward({"input": ua1})
    assert msg1.batch_size == 2
    assert fork.belief.precision_mode == ua1.precision_mode

    # 出力メッセージをキャッシュに保存
    fork.output_message = msg1

    # 2回目 forward: incremental 分岐に入る
    msg2 = fork._compute_forward({"input": ua2})
    assert msg2.batch_size == 2
    assert fork.belief.precision_mode == ua2.precision_mode
    assert not np.allclose(fork.belief.data, ua1.data)


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
