import importlib.util
import pytest
import numpy as np

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse
from gpie.core import backend
from gpie.core.linalg_utils import random_phase_mask
from gpie.core.rng_utils import get_rng

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec is not None:
    import cupy as cp
    backend_libs = [np, cp]
else:
    backend_libs = [np]


@model
def random_structured_cdi(masks, noise):
    obj = ~GaussianPrior(event_shape=(32,32), label = "sample")
    pad_width = ((16,16),(16,16))
    x = obj.zero_pad(pad_width)
    for mask in masks:
        x = fft2(mask * x)
    AmplitudeMeasurement(var = noise, damping = 0.3) << x


@pytest.mark.parametrize("xp", backend_libs)
def test_structured_random_model_reconstruction(xp):
    """Test structured random model reconstruction with numpy/cupy and dtype precision control."""
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    # Generate random phase masks
    n_layers = 2
    phase_masks = [random_phase_mask((64,64), dtype=xp.complex64, rng=rng) for _ in range(n_layers)]

    g = random_structured_cdi(masks = phase_masks, noise = 1e-4)

    sample_wave = g.get_wave("sample")

    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)
    true_sample = sample_wave.get_sample()

    def monitor(graph, t):
        est = graph.get_wave("sample").compute_belief().data
        err = pmse(est, true_sample)
        if t % 20 == 0:
            print(f"[t={t}] PMSE = {err:.5e}")

    g.run(n_iter=100, callback=monitor, verbose=False)

    recon = sample_wave.compute_belief().data
    final_err = pmse(recon, true_sample)
    assert final_err < 1e-3
    assert recon.dtype == xp.complex64
