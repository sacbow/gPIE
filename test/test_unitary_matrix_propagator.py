import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.propagator.unitary_matrix_propagator import UnitaryMatrixPropagator
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_unitary_matrix
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec is not None:
    import cupy as cp
    backend_libs = [np, cp]
else:
    cp = None
    backend_libs = [np]


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagator_to_backend(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    U = random_unitary_matrix(4, rng=rng)
    prop = UnitaryMatrixPropagator(U)

    # simulate batched expansion (as would happen in __matmul__)
    dummy_wave = Wave(event_shape=(4,), batch_size=2)
    _ = prop @ dummy_wave

    if cp is not None and xp.__name__ == "numpy":
        backend.set_backend(cp)
        prop.to_backend()
        assert isinstance(prop.U, cp.ndarray)
        assert isinstance(prop.Uh, cp.ndarray)
        assert prop.U.shape == (2, 4, 4)  # batched
        assert prop.Uh.shape == (2, 4, 4)
        assert prop.Uh.dtype == prop.U.dtype == cp.complex64
    elif cp is None and xp.__name__ == "numpy":
        pytest.skip("CuPy not available, skipping transfer-to-backend test")
    else:
        backend.set_backend(np)
        prop.to_backend()
        assert isinstance(prop.U, np.ndarray)
        assert isinstance(prop.Uh, np.ndarray)
        assert prop.U.shape == (2, 4, 4)
        assert prop.Uh.shape == (2, 4, 4)
        assert prop.Uh.dtype == prop.U.dtype == np.complex64


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagator_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)
    n = 4
    B = 3

    x_wave = Wave(event_shape=(n,), batch_size=B, dtype=xp.complex64, label="x_wave", precision_mode="scalar")
    prop = UnitaryMatrixPropagator(random_unitary_matrix(n, rng=rng))
    prop._set_precision_mode("scalar")
    y_wave = prop @ x_wave
    y_wave._set_precision_mode("scalar")

    class DummyMeasurement:
        def __init__(self):
            self.received = None
        def receive_message(self, wave, message):
            self.received = message

    dummy_meas = DummyMeasurement()
    y_wave.add_child(dummy_meas)

    ua_in = UncertainArray.random(event_shape=(n,), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    ua_out = UncertainArray.random(event_shape=(n,), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)

    prop.receive_message(x_wave, ua_in)
    prop.receive_message(y_wave, ua_out)

    prop.backward()
    assert isinstance(prop.input_messages[x_wave], UncertainArray)
    assert prop.input_messages[x_wave].data.shape == (B, n)

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

    output = UnitaryMatrixPropagator(random_unitary_matrix(n, rng=rng)) @ x_wave
    prop = output.parent

    y_sample = prop.get_sample_for_output(rng)

    assert y_sample.shape == (B, n)
    assert y_sample.dtype == xp.complex64
    x_norm = xp.linalg.norm(x_sample, axis=1)
    y_norm = xp.linalg.norm(y_sample, axis=1)
    assert xp.allclose(x_norm, y_norm, rtol=1e-5)



def test_unitary_propagator_array_to_scalar_compute_belief():
    backend.set_backend(np)
    rng = get_rng(seed=100)
    n, B = 3, 2
    U = random_unitary_matrix(n, rng=rng)
    prop = UnitaryMatrixPropagator(U, precision_mode=UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)

    x_wave = Wave(event_shape=(n,), batch_size=B, dtype=np.complex64)
    y_wave = prop @ x_wave

    ua_in = UncertainArray.random((n,), batch_size=B, dtype=np.complex64, rng=rng, scalar_precision=False)
    ua_out = UncertainArray.random((n,), batch_size=B, dtype=np.complex64, rng=rng, scalar_precision=False)
    prop.receive_message(x_wave, ua_in)
    prop.receive_message(y_wave, ua_out)

    prop.compute_belief()
    assert prop.x_belief is not None
    assert prop.y_belief is not None


def test_unitary_propagator_init_with_3d_and_invalid():
    backend.set_backend(np)
    rng = get_rng(seed=101)
    n, B = 2, 3
    U3d = np.stack([random_unitary_matrix(n, rng=rng) for _ in range(B)], axis=0)
    prop = UnitaryMatrixPropagator(U3d)
    assert prop.U.shape == (B, n, n)

    # Invalid ndim
    with pytest.raises(ValueError):
        _ = UnitaryMatrixPropagator(np.ones((5,), dtype=np.complex64))


def test_unitary_propagator_precision_mode_getters_and_setter():
    backend.set_backend(np)
    U = random_unitary_matrix(2, rng=get_rng(seed=102))

    # SCALAR
    prop = UnitaryMatrixPropagator(U, precision_mode=UnaryPropagatorPrecisionMode.SCALAR)
    assert prop.get_input_precision_mode(None) == PrecisionMode.SCALAR
    assert prop.get_output_precision_mode() == PrecisionMode.SCALAR

    # ARRAY_TO_SCALAR
    prop = UnitaryMatrixPropagator(U, precision_mode=UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
    assert prop.get_input_precision_mode(None) == PrecisionMode.ARRAY
    assert prop.get_output_precision_mode() == PrecisionMode.SCALAR

    # SCALAR_TO_ARRAY
    prop = UnitaryMatrixPropagator(U, precision_mode=UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
    assert prop.get_input_precision_mode(None) == PrecisionMode.SCALAR
    assert prop.get_output_precision_mode() == PrecisionMode.ARRAY

    # Invalid mode string
    with pytest.raises(ValueError):
        prop._set_precision_mode("invalid_mode")

    # Conflict mode
    prop = UnitaryMatrixPropagator(U, precision_mode=UnaryPropagatorPrecisionMode.SCALAR)
    with pytest.raises(ValueError):
        prop._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)


def test_unitary_propagator_matmul_shape_errors():
    backend.set_backend(np)
    U = random_unitary_matrix(3, rng=get_rng(seed=103))
    prop = UnitaryMatrixPropagator(U)

    # Wrong dimensional wave (2D instead of 1D)
    bad_wave = Wave(event_shape=(2, 2), batch_size=1, dtype=np.complex64)
    with pytest.raises(ValueError):
        _ = prop @ bad_wave

    # Shape mismatch between U and wave.event_shape
    wave = Wave(event_shape=(5,), batch_size=1, dtype=np.complex64)
    with pytest.raises(ValueError):
        _ = prop @ wave


def test_unitary_propagator_repr():
    backend.set_backend(np)
    U = random_unitary_matrix(2, rng=get_rng(seed=104))
    prop = UnitaryMatrixPropagator(U, precision_mode=UnaryPropagatorPrecisionMode.SCALAR)
    rep = repr(prop)
    assert "UnitaryMatrixProp" in rep
    assert "mode=" in rep
