# tests/test_blind_ptychography_integration.py

import numpy as np
import pytest
import importlib.util

from gpie import (
    model,
    GaussianPrior,
    GaussianMeasurement,
    fft2,
    mse,
)
from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array


# -------------------------------------------------
# Optional CuPy support
# -------------------------------------------------

cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


# -------------------------------------------------
# Test configuration (easy to tune)
# -------------------------------------------------

OBJ_SHAPE = (64, 64)
PRB_SHAPE = (32, 32)

STRIDE = 16        # <-- overlap control (reduce if unstable)
N_ITER = 30        # modest iteration count for CI
NOISE_VAR = 1e-4

ERROR_THRESHOLD = 1e-2  # PMSE threshold (loose but meaningful)


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def generate_scan_indices(obj_shape, prb_shape, stride):
    """Generate regular grid scan indices."""
    H, W = obj_shape
    h, w = prb_shape

    ys = list(range(0, H - h + 1, stride))
    xs = list(range(0, W - w + 1, stride))

    return [(slice(y, y + h), slice(x, x + w)) for y in ys for x in xs]


# -------------------------------------------------
# Model definition
# -------------------------------------------------

@model
def blind_ptychography_model(indices, noise, dtype):
    """
    Blind ptychography with complex Gaussian observation.
    """
    obj = ~GaussianPrior(
        event_shape=OBJ_SHAPE,
        label="object",
        dtype=dtype,
    )

    prb = ~GaussianPrior(
        event_shape=PRB_SHAPE,
        label="probe",
        dtype=dtype,
    )

    patches = obj.extract_patches(indices)
    exit_waves = prb * patches

    GaussianMeasurement(var=noise) << fft2(exit_waves)
    return


# -------------------------------------------------
# Integration test
# -------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("schedule", ["parallel", "sequential"])
def test_blind_ptychography_end_to_end(xp, schedule):
    """
    End-to-end blind ptychography integration test.

    Checks that:
    - graph builds correctly
    - EP runs without divergence
    - reconstruction error decreases to a reasonable level
    """
    backend.set_backend(xp)

    rng = get_rng(seed=123)
    dtype = xp.complex64

    # -------------------------------------------------
    # Ground truth
    # -------------------------------------------------

    true_obj = random_normal_array(
        (1, *OBJ_SHAPE), dtype=dtype, rng=rng
    )

    true_prb = random_normal_array(
        (1, *PRB_SHAPE), dtype=dtype, rng=rng
    )

    # -------------------------------------------------
    # Scan geometry
    # -------------------------------------------------

    indices = generate_scan_indices(
        OBJ_SHAPE, PRB_SHAPE, STRIDE
    )
    assert len(indices) > 1

    # -------------------------------------------------
    # Build graph
    # -------------------------------------------------

    g = blind_ptychography_model(
        indices=indices,
        noise=NOISE_VAR,
        dtype=dtype,
    )

    g.set_init_rng(get_rng(seed=4))

    # Inject ground truth
    g.get_wave("object").set_sample(true_obj)
    g.get_wave("probe").set_sample(true_prb)

    # Generate synthetic observations
    g.generate_sample(
        rng=get_rng(seed=5),
        update_observed=True,
    )

    # -------------------------------------------------
    # Monitor reconstruction
    # -------------------------------------------------

    history_obj = []
    history_prb = []

    def monitor(graph, t):
        obj_est = graph.get_wave("object").compute_belief().data
        prb_est = graph.get_wave("probe").compute_belief().data

        # normalize to remove scale ambiguity
        obj_est = obj_est / xp.linalg.norm(obj_est)
        prb_est = prb_est / xp.linalg.norm(prb_est)

        obj_true_n = true_obj / xp.linalg.norm(true_obj)
        prb_true_n = true_prb / xp.linalg.norm(true_prb)

        history_obj.append(mse(obj_est, obj_true_n))
        history_prb.append(mse(prb_est, prb_true_n))

    # -------------------------------------------------
    # Run inference
    # -------------------------------------------------

    g.run(
        n_iter=N_ITER,
        schedule=schedule,
        callback=monitor,
    )

    # -------------------------------------------------
    # Assertions (robust & CI-safe)
    # -------------------------------------------------

    assert history_obj[-1] < ERROR_THRESHOLD, (
        f"Object reconstruction failed (schedule={schedule}): "
        f"PMSE={history_obj[-1]:.2e}"
    )

    assert history_prb[-1] < ERROR_THRESHOLD, (
        f"Probe reconstruction failed (schedule={schedule}): "
        f"PMSE={history_prb[-1]:.2e}"
    )

    # Final sanity checks
    obj_est = g.get_wave("object").compute_belief().data
    prb_est = g.get_wave("probe").compute_belief().data

    assert obj_est.shape == (1, *OBJ_SHAPE)
    assert prb_est.shape == (1, *PRB_SHAPE)

    assert isinstance(obj_est, xp.ndarray)
    assert isinstance(prb_est, xp.ndarray)
