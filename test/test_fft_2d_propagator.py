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
from gpie.core.linalg_utils import fft2_centered, ifft2_centered

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_fft2d_propagator_to_backend(xp):
    backend.set_backend(xp)
    prop = FFT2DPropagator(shape=(4, 4), dtype=xp.complex128)

    # Switch backend and test conversion
    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.dtype, type(new_backend.dtype(prop.dtype)))


@pytest.mark.parametrize("xp", backend_libs)
def test_fft2d_propagator_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)
    n = 4

    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x_wave = Wave(shape=(n, n), dtype=xp.complex128, label="x_wave")
            output = FFT2DPropagator() @ x_wave
            with self.observe():
                meas = GaussianMeasurement(var=0.1) @ output
            self.compile()

    g = TestGraph()
    g.set_init_rng(rng)

    x_wave = g.get_wave("x_wave")
    prop = x_wave.children[0]
    output = prop.output

    ua_in = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=True)
    ua_out = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=True)

    prop.receive_message(x_wave, ua_in)
    prop.receive_message(output, ua_out)

    prop.backward()
    assert isinstance(prop.input_messages[x_wave], UncertainArray)

    prop.forward()
    assert isinstance(prop.output_message, UncertainArray)


@pytest.mark.parametrize("xp", backend_libs)
def test_fft2d_propagator_sample_generation(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    n = 4
    x_wave = Wave(shape=(n, n), dtype=xp.complex128)
    x_sample = xp.ones((n, n), dtype=xp.complex128)
    x_wave.set_sample(x_sample)
    output = FFT2DPropagator() @ x_wave
    prop = output.parent

    y_sample = prop.get_sample_for_output(rng)

    assert y_sample.shape == (n, n)
    assert y_sample.dtype == xp.complex128
    assert xp.allclose(xp.linalg.norm(y_sample), xp.linalg.norm(fft2_centered(x_sample)))


@pytest.mark.parametrize("xp", backend_libs)
def test_fft2d_propagator_precision_modes(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    n = 4

    for mode in [
        UnaryPropagatorPrecisionMode.SCALAR,
        UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
        UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
    ]:
        x_wave = Wave(shape=(n, n), dtype=xp.complex128)
        output = FFT2DPropagator() @ x_wave
        prop = output.parent

        ua_in = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=False)
        ua_out = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=False)

        prop.receive_message(x_wave, ua_in)
        prop.receive_message(prop.output, ua_out)

        prop._set_precision_mode(mode)
        prop.compute_belief()

        assert prop.x_belief is not None
        assert prop.y_belief is not None



@pytest.mark.parametrize("xp", backend_libs)
def test_fft2d_propagator_invalid_input_and_errors(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    prop = FFT2DPropagator()

    bad_wave = Wave(shape=(4,), dtype=xp.complex128)
    with pytest.raises(ValueError):
        _ = prop @ bad_wave

    # Test get_sample_for_output raises when no sample is set
    wave = Wave(shape=(4, 4), dtype=xp.complex128)
    output = FFT2DPropagator() @ wave
    prop = output.parent
    with pytest.raises(RuntimeError):
        prop.get_sample_for_output(rng)
