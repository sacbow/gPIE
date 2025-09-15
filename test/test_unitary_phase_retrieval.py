import pytest
import numpy as np
import importlib.util

from gpie import model, SupportPrior, AmplitudeMeasurement
from gpie.graph.propagator.unitary_propagator import UnitaryPropagator
from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array, random_unitary_matrix
from gpie.core.metrics import pmse

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@model
def unitary_observation_model(U, support_mask, var=1e-4, dtype=np.complex64):
    x = ~SupportPrior(event_shape = support_mask.shape, support=support_mask, label="x", dtype=dtype)
    y = UnitaryPropagator(U) @ x
    AmplitudeMeasurement(var=var, damping = 0.3) << y
    return


@pytest.mark.parametrize("xp", backend_libs)
def test_unitary_propagation_reconstruction(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    N = 256  # 1D wave
    shape = (N,)
    dtype = xp.complex64
    noise = 1e-4

    # Generate support: only first 1/3 elements are nonzero
    support_mask = xp.zeros(shape, dtype=bool)
    support_mask[:N // 3] = True

    # True signal
    true_x = random_normal_array(shape, dtype=dtype, rng=rng) * support_mask
    true_x = xp.broadcast_to(true_x, (1, *true_x.shape))

    # Unitary matrix
    U = random_unitary_matrix(N, rng=rng, dtype=xp.complex64)

    # Build graph
    g = unitary_observation_model(U=U, support_mask=support_mask, var=noise, dtype=dtype)

    # Inject known sample
    g.get_wave("x").set_sample(true_x)
    g.set_init_rng(get_rng(seed=11))
    g.generate_sample(rng=get_rng(seed=99), update_observed=True)

    # Inference
    history = []

    def monitor(graph, t):
        est = graph.get_wave("x").compute_belief().data
        err = pmse(est, true_x)
        history.append(err)

    g.run(n_iter=100, callback=monitor, verbose=False)

    # Assertions
    assert g.get_wave("x").children[0].output.precision_mode == "scalar"

    assert history[0] > history[-1]
    assert history[-1] < 1e-3
