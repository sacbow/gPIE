import pytest
import numpy as np
import importlib.util

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, GaussianMeasurement, PhaseMaskPropagator
from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array, random_phase_mask

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@model
def coded_diffraction_model(var, masks, dtype=np.complex64):
    obj = ~GaussianPrior(event_shape=masks[0].shape, label="obj", dtype=dtype)
    for mask in masks:
        AmplitudeMeasurement(var=var, damping = 0.3) << fft2(mask * obj)
    return


@pytest.mark.parametrize("xp", backend_libs)
def test_coded_diffraction_model_reconstruction(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)

    shape = (128,128)
    n_measurements = 3
    dtype = xp.complex64
    noise = 1e-4

    true_obj = random_normal_array((1, *shape), dtype=dtype, rng=rng)
    masks = [random_normal_array(shape, dtype=dtype, rng=rng) for _ in range(n_measurements)]

    # Build graph with @model
    g = coded_diffraction_model(var=noise, masks=masks, dtype=dtype)
    g.set_init_rng(get_rng(seed=4))

    # Inject known sample
    g.get_wave("obj").set_sample(true_obj)
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)

    # Inference
    history = []
    
    def monitor(graph, t):
        x = graph.get_wave("obj").compute_belief().data
        err = pmse(x, true_obj)
        history.append(err)


    g.run(n_iter=200, callback=monitor)
    assert len(g.get_wave("obj").children) == n_measurements
    assert history[-1] < 1e-3

    est = g.get_wave("obj").compute_belief().data
    assert est.shape == (1, *shape)
    assert isinstance(est, xp.ndarray)



