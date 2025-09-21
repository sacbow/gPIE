import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.propagator.unitary_propagator import UnitaryPropagator
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_unitary_matrix
from gpie.core.types import PrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagator_to_backend(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    U = random_unitary_matrix(4, rng=rng)
    prop = UnitaryPropagator(U)

    # simulate batched expansion (as would happen in __matmul__)
    dummy_wave = Wave(event_shape=(4,), batch_size=2)
    _ = prop @ dummy_wave

    new_backend = cp if xp is np else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.U, new_backend.ndarray)
    assert isinstance(prop.Uh, new_backend.ndarray)
    assert prop.U.shape == (2, 4, 4)  # batched
    assert prop.Uh.shape == (2, 4, 4)
    assert prop.Uh.dtype == prop.U.dtype == new_backend.complex64  # default dtype


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagator_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)
    n = 4
    B = 3

    # Setup input wave and propagator manually
    x_wave = Wave(event_shape=(n,), batch_size=B, dtype=xp.complex64, label='x_wave', precision_mode = "scalar")
    prop = UnitaryPropagator(random_unitary_matrix(n, rng=rng))
    prop._set_precision_mode("scalar")
    y_wave = prop @ x_wave
    y_wave._set_precision_mode("scalar")

    # Simulate measurement to create downstream message (minimal dummy)
    class DummyMeasurement:
        def __init__(self):
            self.received = None
        def receive_message(self, wave, message):
            self.received = message

    dummy_meas = DummyMeasurement()
    y_wave.add_child(dummy_meas)

    # Prepare random input/output messages
    ua_in = UncertainArray.random(event_shape=(n,), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    ua_out = UncertainArray.random(event_shape=(n,), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)

    # Inject messages manually
    prop.receive_message(x_wave, ua_in)
    prop.receive_message(y_wave, ua_out)

    # Backward pass updates input message
    prop.backward()
    assert isinstance(prop.input_messages[x_wave], UncertainArray)
    assert prop.input_messages[x_wave].data.shape == (B, n)

    # Forward pass sends message to child
    prop.forward()
    y_wave.forward()
    assert isinstance(dummy_meas.received, UncertainArray)
    assert dummy_meas.received.data.shape == (B, n)


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagator_sample_generation(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    n = 4
    B = 2

    x_wave = Wave(event_shape=(n,), batch_size=B, dtype=xp.complex64)
    x_sample = xp.ones((B, n), dtype=xp.complex64)
    x_wave.set_sample(x_sample)

    output = UnitaryPropagator(random_unitary_matrix(n, rng=rng)) @ x_wave
    prop = output.parent

    y_sample = prop.get_sample_for_output(rng)

    assert y_sample.shape == (B, n)
    assert y_sample.dtype == xp.complex64
    x_norm = xp.linalg.norm(x_sample, axis=1)
    y_norm = xp.linalg.norm(y_sample, axis=1)
    assert xp.allclose(x_norm, y_norm, rtol=1e-5)

