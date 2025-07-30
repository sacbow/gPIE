import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.propagator.phase_mask_propagator import PhaseMaskPropagator
from gpie.graph.wave import Wave
from gpie.graph.structure.graph import Graph
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
from gpie.core.uncertain_array import UncertainArray
from gpie.core.linalg_utils import random_phase_mask
from gpie.core.rng_utils import get_rng

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_phase_mask_propagator_forward_backward(xp):
    """Test forward/backward passes for PhaseMaskPropagator."""
    backend.set_backend(xp)
    rng = get_rng(seed=0)
    n = 4
    phase_mask = random_phase_mask((n, n), dtype=xp.complex128, rng=rng)

    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x_wave = Wave(shape=(n, n), dtype=xp.complex128, label="x_wave")
            output = PhaseMaskPropagator(phase_mask) @ x_wave
            with self.observe():
                GaussianMeasurement(var=0.1) @ output
            self.compile()

    g = TestGraph()
    g.set_init_rng(rng)

    # Retrieve nodes
    x_wave = g.get_wave("x_wave")
    prop = x_wave.children[0]
    output = prop.output

    # Inject random messages
    ua_in = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=True)
    ua_out = UncertainArray.random((n, n), dtype=xp.complex128, rng=rng, scalar_precision=True)
    prop.receive_message(x_wave, ua_in)
    prop.receive_message(output, ua_out)

    # Backward pass
    prop.backward()
    assert isinstance(prop.input_messages[x_wave], UncertainArray)

    # Forward pass
    prop.forward()
    assert isinstance(prop.output_message, UncertainArray)


@pytest.mark.parametrize("xp", backend_libs)
def test_phase_mask_propagator_to_backend(xp):
    """Test that phase_mask is properly transferred between backends."""
    backend.set_backend(xp)
    rng = get_rng(seed=1)
    n = 4
    phase_mask = random_phase_mask((n, n), dtype=xp.complex128, rng=rng)
    prop = PhaseMaskPropagator(phase_mask)

    # Switch backend and transfer
    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.phase_mask, new_backend.ndarray)
    assert prop.phase_mask.dtype == prop.dtype
    assert new_backend.allclose(new_backend.abs(prop.phase_mask), 1.0)

@pytest.mark.parametrize("xp", backend_libs)
def test_phase_mask_propagator_get_sample_for_output(xp):
    """Test sample retrieval using get_sample_for_output."""
    backend.set_backend(xp)
    rng = get_rng(seed=2)
    n = 4
    phase_mask = random_phase_mask((n, n), dtype=xp.complex128, rng=rng)

    # Create input wave with a fixed sample
    x_wave = Wave(shape=(n, n), dtype=xp.complex128)
    sample = xp.ones((n, n), dtype=xp.complex128)
    x_wave.set_sample(sample)

    # Build propagator and connect wave
    output = PhaseMaskPropagator(phase_mask) @ x_wave
    prop = output.parent

    # Retrieve propagated sample via get_sample_for_output
    y_sample = prop.get_sample_for_output(rng)
    assert xp.allclose(y_sample, sample * phase_mask)

    # Also confirm that output.set_sample can accept this result
    output.set_sample(y_sample)
    assert xp.allclose(output.get_sample(), sample * phase_mask)


@pytest.mark.parametrize("xp", backend_libs)
def test_phase_mask_propagator_invalid_inputs(xp):
    """Test invalid phase_mask and input shape errors."""
    backend.set_backend(xp)
    rng = get_rng(seed=3)
    n = 4

    # Non-unit magnitude mask
    bad_mask = xp.ones((n, n), dtype=xp.complex128) * 2.0
    with pytest.raises(ValueError):
        PhaseMaskPropagator(bad_mask)

    # Shape mismatch
    phase_mask = random_phase_mask((n, n), dtype=xp.complex128, rng=rng)
    prop = PhaseMaskPropagator(phase_mask)
    bad_wave = Wave(shape=(n + 1, n), dtype=xp.complex128)
    with pytest.raises(ValueError):
        _ = prop @ bad_wave
