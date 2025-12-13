import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.structure.graph import Graph
from gpie.graph.wave import Wave
from gpie.graph.prior.gaussian_prior import GaussianPrior
from gpie.graph.propagator.unitary_matrix_propagator import UnitaryMatrixPropagator
from gpie.graph.measurement.gaussian_measurement import GaussianMeasurement
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_unitary_matrix

cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_graph_generate_sample_updates_waves_and_measurements(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    class TestGraph(Graph):
        def __init__(self):
            super().__init__()
            x = ~GaussianPrior(event_shape = (4,))
            y = UnitaryMatrixPropagator(random_unitary_matrix(4, rng=rng)) @ x
            with self.observe():
                meas = GaussianMeasurement(var=0.1) << y
            self.compile()

    g = TestGraph()

    # Before sample generation: waves and measurements have no samples
    for wave in g._waves:
        assert wave.get_sample() is None

    for factor in g._factors:
        if hasattr(factor, "observed"):
            assert factor.observed is None

    # Generate samples
    g.generate_sample(rng=rng, update_observed=True)

    # After generation: waves should now have samples
    for wave in g._waves:
        assert wave.get_sample() is not None

    # Measurements should also be updated
    for factor in g._factors:
        if hasattr(factor, "observed"):
            assert factor.observed is not None
            assert factor.observed.event_shape == wave.event_shape  # matching shape