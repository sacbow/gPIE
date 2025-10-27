import importlib.util
import pytest
import numpy as np

from gpie import model, SupportPrior, fft2, AmplitudeMeasurement, mse
from gpie.core import backend
from gpie.core.linalg_utils import circular_aperture, masked_random_array
from gpie.core.rng_utils import get_rng

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec is not None:
    import cupy as cp
    backend_libs = [np, cp]
else:
    backend_libs = [np]


@model
def holography_model(var, ref_wave, support, dtype=np.complex64):
    """gPIE holography model with DSL-based graph definition."""
    obj = ~SupportPrior(event_shape=ref_wave.shape, support=support, label="obj", dtype=dtype)
    AmplitudeMeasurement(var=var, damping = 0.3) << fft2(ref_wave + obj)
    return


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("obj_dtype", [np.complex128, np.complex64])
def test_holography_reconstruction(xp, obj_dtype):
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    H, W = 128, 128
    shape = (H, W)
    noise = 1e-4

    support_x = circular_aperture(shape, radius=0.2, center=(-0.2, -0.2))
    support_y = circular_aperture(shape, radius=0.2, center=(0.2, 0.2))
    ref_wave = masked_random_array(support_x, dtype=xp.complex128, rng=rng)

    g = holography_model(var=noise, ref_wave=ref_wave, support=support_y, dtype=obj_dtype)
    g.set_init_rng(get_rng(seed=11))
    g.generate_sample(rng=get_rng(seed=9), update_observed=True)

    true_obj = g.get_wave("obj").get_sample()

    def monitor(graph, t):
        x = graph.get_wave("obj").compute_belief().data
        err = mse(x, true_obj)
        if t % 10 == 0:
            print(f"[t={t}] MSE = {err:.5e}")

    g.run(n_iter=50, callback=monitor, verbose=False)

    recon = g.get_wave("obj").compute_belief().data
    final_err = mse(recon, true_obj)
    assert final_err < 1e-3
    assert recon.dtype == xp.dtype(obj_dtype)


@pytest.mark.parametrize("xp", backend_libs)
def test_holography_to_backend(xp):
    backend.set_backend(np)
    rng = get_rng(seed=0)

    H, W = 32, 32
    shape = (H, W)
    support = circular_aperture(shape=shape, radius=0.3)
    ref_wave = masked_random_array(support, dtype=np.complex128, rng=rng)

    g = holography_model(var=1e-4, ref_wave=ref_wave, support=support)

    # GPU transfer (only if cupy available)
    if xp.__name__ == "cupy":
        backend.set_backend(cp)
        g.to_backend()
        for w in g._waves:
            assert w.dtype == cp.complex64

    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)
    g.run(n_iter=5, verbose=False)
