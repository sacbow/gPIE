import pytest
import numpy as np
import importlib.util

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse, replicate
from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array

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
    """
    Coded diffraction pattern model using ForkPropagator via replicate().

    Args:
        var (float): Noise variance.
        masks (ndarray): Shape (B, H, W), batch of random phase masks.
        dtype: Complex dtype.
    """
    B, H, W = masks.shape

    # Prior object (single sample, batch_size=1)
    obj = ~GaussianPrior(event_shape=(H, W), label="obj", dtype=dtype)

    # Replicate across batch dimension
    obj_batch = replicate(obj, batch_size=B)

    # Apply masks (batched elementwise multiplication)
    masked = masks * obj_batch

    # FFT
    Y = fft2(masked)

    # Amplitude measurement (batched)
    AmplitudeMeasurement(var=var, damping=0.2) << Y
    return


@pytest.mark.parametrize("xp", backend_libs)
def test_coded_diffraction_model_reconstruction(xp):
    backend.set_backend(xp)
    rng = get_rng(seed=123)

    shape = (64, 64)
    n_measurements = 4
    dtype = xp.complex64
    noise = 1e-4

    # ground-truth object
    true_obj = random_normal_array((1, *shape), dtype=dtype, rng=rng)

    # batched random masks
    masks = random_normal_array((n_measurements, *shape), dtype=dtype, rng=rng)

    # Build graph with @model
    g = coded_diffraction_model(var=noise, masks=masks, dtype=dtype)
    g.set_init_rng(get_rng(seed=4))

    # Inject known sample
    g.get_wave("obj").set_sample(true_obj)
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)

    # Inference with monitoring
    history = []

    def monitor(graph, t):
        x = graph.get_wave("obj").compute_belief().data
        err = pmse(x, true_obj)
        history.append(err)

    g.run(n_iter=100, callback=monitor)

    # Check convergence
    assert history[-1] < 1e-3

    est = g.get_wave("obj").compute_belief().data
    assert est.shape == (1, *shape)
    assert isinstance(est, xp.ndarray)
