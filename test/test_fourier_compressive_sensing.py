import importlib.util
import pytest
import numpy as np

import gpie
from gpie import model, observe, mse, fft2, Graph
from gpie import SparsePrior, GaussianMeasurement
from gpie.core.linalg_utils import random_binary_mask, random_phase_mask
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
def test_fft_compressive_sensing_reconstruction(xp):
    """Test compressive sensing using fft-based propagator."""
    gpie.set_backend(xp)

    # --- Parameters ---
    event_shape = (128,128)
    rho = 0.1
    var = 1e-4
    mask_ratio = 0.3

    rng = get_rng(seed=42)
    mask = random_binary_mask(event_shape, subsampling_rate=mask_ratio, rng=rng)

    # --- Model Definition ---
    class FFTCSGraph(Graph):
        def __init__(self, var, rho, event_shape, dtype):
            super().__init__()
            x = ~SparsePrior(rho=rho, event_shape=event_shape, damping=0.03, label="x", dtype=dtype)
            with self.observe():
                GaussianMeasurement(var=var, with_mask=True) << fft2(x)
            self.compile()

    g = FFTCSGraph(var=var, rho=rho, event_shape=event_shape, dtype=xp.complex64)


    # --- Initialization and Sampling ---
    g.set_init_rng(get_rng(seed=11))
    g.generate_sample(rng=get_rng(seed=123), mask=None)

    true_x = g.get_wave("x").get_sample()

    # --- Inference ---
    mse_list = []

    def monitor(graph, t):
        est = graph.get_wave("x").compute_belief().data
        mse_list.append(mse(est, true_x))

    g.run(n_iter=50, callback=monitor, verbose=False)

    # --- Assertions ---
    assert mse_list[-1] < 1e-4, f"Final MSE too high: {mse_list[-1]:.2e}"
