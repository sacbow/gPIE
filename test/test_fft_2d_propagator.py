import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.propagator.fft_2d_propagator import FFT2DPropagator
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
from gpie.graph.wave import Wave
from gpie.graph.structure.graph import Graph
from gpie.core.uncertain_array import UncertainArray
from gpie.core.rng_utils import get_rng
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode
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
def test_fft2d_to_backend(xp):
    backend.set_backend(xp)
    prop = FFT2DPropagator(event_shape=(8, 8), dtype=xp.complex64)

    # Change backend and test dtype conversion
    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.dtype, type(new_backend.dtype(prop.dtype)))


@pytest.mark.parametrize("xp", backend_libs)
def test_fft2d_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    n = 8
    B = 4  # batch size

    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64, label="x")
            y = FFT2DPropagator() @ x
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
def test_fft2d_sample_generation(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)
    n = 16
    B = 2

    x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64)
    x_sample = xp.ones((B, n, n), dtype=xp.complex64)
    x.set_sample(x_sample)

    y = FFT2DPropagator() @ x
    prop = y.parent
    y_sample = prop.get_sample_for_output(rng)

    assert y_sample.shape == (B, n, n)
    assert y_sample.dtype == xp.complex64
    expected = get_fft_backend().fft2_centered(x_sample)
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
        y = FFT2DPropagator() @ x
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
def test_fft2d_invalid_cases(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=999)

    # invalid input shape (not 2D)
    prop = FFT2DPropagator()
    bad_wave = Wave(event_shape=(4,), dtype=xp.complex64)
    with pytest.raises(ValueError):
        _ = prop @ bad_wave

    # sample not set
    wave = Wave(event_shape=(4, 4), dtype=xp.complex64)
    out = FFT2DPropagator() @ wave
    prop = out.parent
    with pytest.raises(RuntimeError):
        prop.get_sample_for_output(rng)
