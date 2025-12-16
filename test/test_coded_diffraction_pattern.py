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

    # Replicate across batch dimension (ForkPropagator)
    obj_batch = replicate(obj, batch_size=B)

    # Apply masks (MultiplyConstPropagator)
    masked = masks * obj_batch

    # FFT (FFT2DPropagator)
    Y = fft2(masked)

    # Amplitude measurement
    AmplitudeMeasurement(var=var, damping=0.3) << Y
    return


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_coded_diffraction_model_reconstruction(xp, schedule):
    """
    Verify that coded diffraction pattern reconstruction converges
    for both parallel and sequential schedules.
    """
    backend.set_backend(xp)
    rng = get_rng(seed=123)

    shape = (64, 64)
    n_measurements = 4
    dtype = xp.complex64
    noise = 1e-4

    # Ground-truth object (batch_size=1)
    true_obj = random_normal_array((1, *shape), dtype=dtype, rng=rng)

    # Batched random masks
    masks = random_normal_array((n_measurements, *shape), dtype=dtype, rng=rng)

    # Build graph
    g = coded_diffraction_model(var=noise, masks=masks, dtype=dtype)
    g.set_init_rng(get_rng(seed=4))

    # Inject known sample and generate observations
    g.get_wave("obj").set_sample(true_obj)
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)

    # Track reconstruction error
    history = []

    def monitor(graph, t):
        x = graph.get_wave("obj").compute_belief().data
        err = pmse(x, true_obj)
        history.append(err)

    # Run inference
    g.run(
        n_iter=100,
        schedule=schedule,
        callback=monitor,
    )

    # Convergence check
    assert history[-1] < 1e-3, (
        f"CDP reconstruction did not converge under schedule='{schedule}': "
        f"final PMSE={history[-1]:.2e}"
    )

    # Final estimate sanity check
    est = g.get_wave("obj").compute_belief().data
    assert est.shape == (1, *shape)
    assert isinstance(est, xp.ndarray)
