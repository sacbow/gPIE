import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.propagator.unitary_propagator import UnitaryPropagator
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
from gpie.graph.wave import Wave
from gpie.graph.structure.graph import Graph
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

    # Switch backend and test conversion
    new_backend = cp if xp is np else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.U, new_backend.ndarray)
    assert isinstance(prop.Uh, new_backend.ndarray)
    assert prop.U.shape == (4, 4)
    assert prop.Uh.shape == (4, 4)
    assert prop.Uh.dtype == prop.U.dtype == new_backend.complex128


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagator_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)
    n = 4
    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x_wave = Wave(shape=(n,), dtype=xp.complex128, label = 'x_wave')
            output = UnitaryPropagator(random_unitary_matrix(n, rng=rng)) @ x_wave
            with self.observe():
                meas = GaussianMeasurement(var = 0.1) @ output
            self.compile()

    g = TestGraph()
    g.set_init_rng(get_rng(seed = 9))

    # Provide random messages to input and output
    rng = get_rng(seed = 11)
    ua_in = UncertainArray.random((n,), dtype=xp.complex128, rng=rng, scalar_precision=True)
    ua_out = UncertainArray.random((n,), dtype=xp.complex128, rng=rng, scalar_precision=True)

    #forward/backward
    x_wave = g.get_wave('x_wave')
    prop = x_wave.children[0]
    output = prop.output
    prop.receive_message(x_wave, ua_in)  # simulate input message reception
    prop.receive_message(output, ua_out)  # simulate output message reception


    # Backward pass should update input message
    prop.backward()
    assert isinstance(prop.input_messages[x_wave], UncertainArray)

    # Forward pass should update output wave
    prop.forward()
    assert isinstance(prop.output_message, UncertainArray)


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagator_sample_generation(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)
    n = 4
    x_wave = Wave(shape=(n,), dtype=xp.complex128)
    x_sample = xp.ones((n,), dtype=xp.complex128)
    x_wave.set_sample(x_sample)
    output = UnitaryPropagator(random_unitary_matrix(n, rng=rng)) @ x_wave
    prop = output.parent

    y_sample = prop.get_sample_for_output(rng)

    assert y_sample.shape == (n,)
    assert y_sample.dtype == xp.complex128
    assert xp.allclose(xp.linalg.norm(y_sample), xp.linalg.norm(x_sample))

