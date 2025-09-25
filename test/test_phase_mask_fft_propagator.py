import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.propagator.phase_mask_fft_propagator import PhaseMaskFFTPropagator
from gpie.graph.wave import Wave
from gpie.graph.structure.graph import Graph
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
from gpie.core.uncertain_array import UncertainArray
from gpie.core.linalg_utils import random_phase_mask
from gpie.core.rng_utils import get_rng
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
def test_phase_mask_fft_forward_backward(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    n, B = 4, 3
    phase_mask = random_phase_mask((n, n), dtype=xp.complex64, rng=rng)

    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64, label="x")
            y = PhaseMaskFFTPropagator(phase_mask) @ x
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
def test_phase_mask_fft_to_backend(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=1)
    n = 4
    phase_mask = random_phase_mask((n, n), dtype=xp.complex64, rng=rng)
    prop = PhaseMaskFFTPropagator(phase_mask)

    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.phase_mask, new_backend.ndarray)
    assert isinstance(prop.phase_mask_conj, new_backend.ndarray)
    assert new_backend.allclose(new_backend.abs(prop.phase_mask), 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_phase_mask_fft_get_sample_for_output(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=2)
    n, B = 6, 2
    phase_mask = random_phase_mask((n, n), dtype=xp.complex64, rng=rng)

    x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64)
    x_sample = xp.ones((B, n, n), dtype=xp.complex64)
    x.set_sample(x_sample)

    y = PhaseMaskFFTPropagator(phase_mask) @ x
    prop = y.parent
    y_sample = prop.get_sample_for_output(rng)
    fft = get_fft_backend()
    expected = fft.ifft2_centered(phase_mask * fft.fft2_centered(x_sample))
    assert xp.allclose(y_sample, expected)


@pytest.mark.parametrize("xp", backend_libs)
def test_phase_mask_fft_invalid_inputs(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=3)
    n = 4

    # Invalid: magnitude ≠ 1
    bad_mask = xp.ones((n, n), dtype=xp.complex64) * 2.0
    with pytest.raises(ValueError):
        PhaseMaskFFTPropagator(bad_mask)

    # Invalid: event_shape mismatch
    phase_mask = random_phase_mask((n, n), dtype=xp.complex64, rng=rng)
    prop = PhaseMaskFFTPropagator(phase_mask)
    bad_wave = Wave(event_shape=(n+1, n), dtype=xp.complex64)
    with pytest.raises(ValueError):
        _ = prop @ bad_wave


@pytest.mark.parametrize("xp", backend_libs)
def test_phase_mask_fft_batch_mask_handling(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=4)
    n, B = 5, 4

    # 3D mask: batch-wise modulation
    phase_mask = random_phase_mask((B, n, n), dtype=xp.complex64, rng=rng)
    x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64)
    y = PhaseMaskFFTPropagator(phase_mask) @ x
    assert y.batch_size == B
    assert y.event_shape == (n, n)

    # 2D mask: shared across batch (should broadcast)
    phase_mask_shared = random_phase_mask((n, n), dtype=xp.complex64, rng=rng)
    x = Wave(event_shape=(n, n), batch_size=B, dtype=xp.complex64)
    y = PhaseMaskFFTPropagator(phase_mask_shared) @ x
    assert y.batch_size == B
    assert y.event_shape == (n, n)
