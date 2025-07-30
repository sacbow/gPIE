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
from gpie.core.types import UnaryPropagatorPrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_propagator_to_backend(xp):
    backend.set_backend(xp)
    prop = IFFT2DPropagator(shape=(4, 4), dtype=xp.complex128)

    # Switch backend and test dtype sync
    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()
    assert prop.dtype == new_backend.complex128


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_propagator_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)
    n = 4

    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x_wave = Wave(shape=(n, n), dtype=xp.complex128, label="x_wave")
            output = IFFT2DPropagator() @ x_wave
            with self.observe():
                meas = GaussianMeasurement(var=0.1) @ output
            self.compile()

    g = TestGraph()
    g.set_init_rng(rng)

    # Fetch nodes
    x_wave = g.get_wave("x_wave")
    prop = x_wave.children[0]
    output = prop.output

    # Prepare synthetic messages
    ua_in = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=True)
    ua_out = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=True)

    # Inject messages manually
    prop.receive_message(x_wave, ua_in)
    prop.receive_message(output, ua_out)

    # Backward pass
    prop.backward()
    assert isinstance(prop.input_messages[x_wave], UncertainArray)

    # Forward pass
    prop.forward()
    assert isinstance(prop.output_message, UncertainArray)



@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_propagator_sample_generation(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    n = 4
    x_wave = Wave(shape=(n, n), dtype=xp.complex128)
    x_wave.set_sample(xp.ones((n, n), dtype=xp.complex128))

    output = IFFT2DPropagator() @ x_wave
    prop = output.parent
    y_sample = prop.get_sample_for_output(rng)

    assert y_sample.shape == (n, n)
    assert xp.allclose(xp.linalg.norm(y_sample), xp.linalg.norm(x_wave.get_sample()))


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_propagator_precision_modes(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    n = 4

    x_wave = Wave(shape=(n, n), dtype=xp.complex128)
    output = IFFT2DPropagator() @ x_wave
    prop = output.parent

    ua_in = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=False)
    ua_out = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=False)

    prop.receive_message(x_wave, ua_in)
    prop.receive_message(prop.output, ua_out)

    # Test multiple precision modes independently
    for mode in [
        UnaryPropagatorPrecisionMode.SCALAR,
        UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY,
        UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR,
    ]:
        new_prop = IFFT2DPropagator() @ x_wave
        new_prop = new_prop.parent
        new_prop.receive_message(x_wave, ua_in)
        new_prop.receive_message(new_prop.output, ua_out)
        new_prop._set_precision_mode(mode)
        new_prop.compute_belief()
        assert new_prop.x_belief is not None
        assert new_prop.y_belief is not None


@pytest.mark.parametrize("xp", backend_libs)
def test_ifft2d_propagator_matmul_dtype_inference(xp):
    backend.set_backend(xp)
    wave = Wave(shape=(4, 4), dtype=xp.float32)
    output = IFFT2DPropagator() @ wave
    prop = output.parent
    assert prop.dtype == xp.complex64
    assert output.dtype == xp.complex64


def test_ifft2d_propagator_repr():
    prop = IFFT2DPropagator(shape=(4, 4))
    r = repr(prop)
    assert "IFFT2DProp" in r
    assert "mode=" in r
