import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.propagator.ifft_2d_propagator import IFFT2DPropagator
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
from gpie.graph.structure.graph import Graph
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.rng_utils import get_rng
from gpie.core.types import UnaryPropagatorPrecisionMode, PrecisionMode
from gpie.core.fft import get_fft_backend

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_to_backend(xp):
    backend.set_backend(xp)
    prop = IFFT2DPropagator(event_shape=(4, 4), dtype=xp.complex64)

    # Change backend and test dtype conversion
    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()
    assert isinstance(prop.dtype, type(new_backend.dtype(prop.dtype)))


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)
    n, B = 8, 3

    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64, label="x")
            y = IFFT2DPropagator() @ x
            with self.observe():
                GaussianMeasurement(var=0.1) << y
            self.compile()

    g = TestGraph()
    g.set_init_rng(rng)

    x = g.get_wave("x")
    prop = x.children[0]
    y = prop.output

    ua_in = UncertainArray.random((n, n), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    ua_out = UncertainArray.random((n, n), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)

    prop.receive_message(x, ua_in)
    prop.receive_message(y, ua_out)

    prop.backward()
    assert isinstance(prop.input_messages[x], UncertainArray)

    prop.forward()
    assert isinstance(prop.output_message, UncertainArray)


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_sample_generation(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    n, B = 16, 2
    x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64)
    x_sample = xp.ones((B, n, n), dtype=xp.complex64)
    x.set_sample(x_sample)

    y = IFFT2DPropagator() @ x
    prop = y.parent
    y_sample = prop.get_sample_for_output(rng)

    assert y_sample.shape == (B, n, n)
    assert y_sample.dtype == xp.complex64
    expected = get_fft_backend().ifft2_centered(x_sample)
    assert xp.allclose(y_sample, expected)


@pytest.mark.parametrize("xp", backend_libs)
def test_fft2d_precision_modes(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=2025)
    n = 8
    B = 3

    for mode in [
        UnaryPropagatorPrecisionMode.SCALAR,
        UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
        UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
    ]:
        x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64)
        y = IFFT2DPropagator() @ x
        prop = y.parent

        ua_in = UncertainArray.random((n, n), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=(mode != UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR))
        ua_out = UncertainArray.random((n, n), batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=(mode != UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY))

        prop.receive_message(x, ua_in)
        prop.receive_message(y, ua_out)

        prop._set_precision_mode(mode)
        prop.compute_belief()

        assert prop.x_belief is not None
        assert prop.y_belief is not None
        assert prop.x_belief.batch_size == B
        assert prop.y_belief.batch_size == B


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_matmul_dtype_inference(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), dtype=xp.float32)
    output = IFFT2DPropagator() @ wave
    prop = output.parent
    assert prop.dtype == xp.complex64
    assert output.dtype == xp.complex64


def test_ifft2d_repr():
    prop = IFFT2DPropagator(event_shape=(8, 8))
    r = repr(prop)
    assert "IFFT2DProp" in r
    assert "mode=" in r


def test_ifft2d_precision_mode_getters():
    backend.set_backend(np)
    n = 4
    x = Wave(event_shape=(n, n), batch_size=1, dtype=np.complex64)

    # SCALAR
    prop = IFFT2DPropagator(event_shape=(n, n), precision_mode=UnaryPropagatorPrecisionMode.SCALAR)
    y = prop @ x
    assert prop.get_input_precision_mode(x) == PrecisionMode.SCALAR
    assert prop.get_output_precision_mode() == PrecisionMode.SCALAR

    # SCALAR_TO_ARRAY
    prop = IFFT2DPropagator(event_shape=(n, n), precision_mode=UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
    y = prop @ x
    assert prop.get_input_precision_mode(x) == PrecisionMode.SCALAR
    assert prop.get_output_precision_mode() == PrecisionMode.ARRAY

    # ARRAY_TO_SCALAR
    prop = IFFT2DPropagator(event_shape=(n, n), precision_mode=UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
    y = prop @ x
    assert prop.get_input_precision_mode(x) == PrecisionMode.ARRAY
    assert prop.get_output_precision_mode() == PrecisionMode.SCALAR
