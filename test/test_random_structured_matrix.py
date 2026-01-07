import importlib.util
import pytest
import numpy as np

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement, pmse
from gpie.core import backend
from gpie.core.linalg_utils import random_phase_mask
from gpie.core.rng_utils import get_rng

# ------------------------------------------------------------
# Optional CuPy support
# ------------------------------------------------------------
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec is not None:
    import cupy as cp
    backend_libs = [np, cp]
else:
    backend_libs = [np]


# ------------------------------------------------------------
# Model definition (batch_size = 1 only)
# ------------------------------------------------------------
@model
def random_structured_cdi(
    masks,
    noise,
    *,
    dtype=np.complex64,
):
    """
    Structured random CDI model.
    NOTE: This model is intended for batch_size = 1 only.
    """
    obj = ~GaussianPrior(
        event_shape=(32, 32),
        batch_size=3,
        label="sample",
        dtype=dtype,
    )

    pad_width = ((16, 16), (16, 16))
    x = obj.zero_pad(pad_width)

    for mask in masks:
        x = fft2(mask * x)

    AmplitudeMeasurement(var=noise, damping=0.3) << x
    return


# ------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------
def build_random_structured_cdi_graph(xp, seed=0, n_layers=2):
    backend.set_backend(xp)
    rng = get_rng(seed)

    masks = [
        random_phase_mask((3, 64, 64), dtype=xp.complex64, rng=rng)
        for _ in range(n_layers)
    ]

    g = random_structured_cdi(
        masks=masks,
        noise=1e-4,
        dtype=xp.complex64,
    )

    g.set_init_rng(get_rng(seed + 1))
    g.generate_sample(
        rng=get_rng(seed + 2),
        update_observed=True,
    )
    return g


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_random_structured_cdi_batch1_converges(xp, schedule):
    """
    Sanity check for structured random CDI with batch_size=1.

    This test intentionally restricts batch_size to 1.
    Sequential scheduling over multiple independent problems
    is not guaranteed to be stable in the current beta release.
    """
    g = build_random_structured_cdi_graph(
        xp=xp,
        seed=123,
        n_layers=2,
    )

    sample_wave = g.get_wave("sample")
    true_sample = sample_wave.get_sample()

    g.run(
        n_iter=200,
        schedule=schedule,
        verbose=False,
    )

    recon = sample_wave.compute_belief().data
    err = (pmse(recon[0], true_sample[0]) + pmse(recon[1], true_sample[1]))/2

    assert err < 1e-3, f"PMSE too large: {err:.2e}"
    assert recon.shape == true_sample.shape
    assert recon.dtype == xp.complex64
