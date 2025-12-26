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


# ------------------------------------------------------------
# Model definition
# ------------------------------------------------------------

@model
def structured_random_cdi(
    masks,
    noise,
    *,
    batch_size: int,
    dtype=np.complex64,
):
    """
    Structured CDI model with explicit batch_size and zero-padding.
    """
    obj = ~GaussianPrior(
        event_shape=(32, 32),
        batch_size=batch_size,
        label="sample",
        dtype=dtype,
    )

    pad_width = ((16, 16), (16, 16))
    x = obj.zero_pad(pad_width)

    for mask in masks:
        x = fft2(mask * x)

    AmplitudeMeasurement(var=noise, damping = 0.3) << x
    return


# ------------------------------------------------------------
# Test utility
# ------------------------------------------------------------

def build_structured_cdi_graph(
    xp,
    batch_size,
    n_layers=2,
    seed=0,
):
    backend.set_backend(xp)
    rng = get_rng(seed)

    masks = [
        random_phase_mask((batch_size, 64, 64), dtype=xp.complex64, rng=rng)
        for _ in range(n_layers)
    ]

    g = structured_random_cdi(
        masks=masks,
        noise=1e-4,
        batch_size=batch_size,
        dtype=xp.complex64,
    )

    g.set_init_rng(get_rng(seed + 1))
    g.generate_sample(rng=get_rng(seed + 2), update_observed=True)
    return g


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_structured_random_cdi_parallel_sequential(xp, schedule):
    """
    Sanity check: structured CDI with zero_pad converges under
    both parallel and sequential scheduling.
    """
    g = build_structured_cdi_graph(
        xp=xp,
        batch_size=2,
        seed=99,
    )

    sample_wave = g.get_wave("sample")
    true_sample = sample_wave.get_sample()

    g.run(
        n_iter=200,
        schedule=schedule,
        verbose=False,
    )

    recon = sample_wave.compute_belief().data
    err = pmse(recon[0], true_sample[0])

    assert err < 1e-3
    assert recon.shape == true_sample.shape
    assert recon.dtype == xp.complex64
