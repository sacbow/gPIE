# test/test_metric.py
import pytest
import numpy as np
from gpie.core import metrics
from gpie.core.backend import set_backend, get_backend

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.mark.parametrize("backend", ["numpy", "cupy"] if HAS_CUPY else ["numpy"])
def test_mse_nmse_psnr(backend):
    # --- Set backend ---
    if backend == "numpy":
        set_backend(np)
    elif backend == "cupy":
        set_backend(cp)
    xp = get_backend()

    x_true = xp.array([1.0, 2.0, 3.0])
    x_est = xp.array([1.1, 1.9, 2.8])

    # MSE
    mse_val = metrics.mse(x_est, x_true)
    assert xp.isclose(mse_val, xp.mean(xp.abs(x_est - x_true) ** 2))

    # NMSE
    nmse_val = metrics.nmse(x_est, x_true)
    expected_nmse = (xp.linalg.norm(x_est - x_true) ** 2) / (xp.linalg.norm(x_true) ** 2)
    assert xp.isclose(nmse_val, expected_nmse)

    # PSNR (finite case)
    psnr_val = metrics.psnr(x_est, x_true, max_val=1.0)
    expected_psnr = 10 * xp.log10(1.0 ** 2 / mse_val)
    assert xp.isclose(psnr_val, expected_psnr)

    # PSNR (zero MSE → inf)
    psnr_inf = metrics.psnr(x_true, x_true, max_val=1.0)
    assert psnr_inf == float("inf")


@pytest.mark.parametrize("backend", ["numpy", "cupy"] if HAS_CUPY else ["numpy"])
def test_support_error(backend):
    if backend == "numpy":
        set_backend(np)
    elif backend == "cupy":
        set_backend(cp)
    xp = get_backend()

    x_true = xp.array([1.0, 0.0, 2.0, 0.0])
    x_est = xp.array([0.9, 0.1, 0.0, 0.0])  # mismatch on indices 1 and 2

    err = metrics.support_error(x_est, x_true, threshold=0.05)
    assert xp.isclose(err, 0.5)  # 2 mismatches out of 4 → 0.5


@pytest.mark.parametrize("backend", ["numpy", "cupy"] if HAS_CUPY else ["numpy"])
def test_phase_align_and_phase_metrics(backend):
    if backend == "numpy":
        set_backend(np)
    elif backend == "cupy":
        set_backend(cp)
    xp = get_backend()

    # True signal and phase-shifted estimate
    x_true = xp.array([1+1j, 2+2j, 3+3j])
    phase_shift = xp.exp(1j * xp.pi / 4)  # 45° phase
    x_est = x_true * phase_shift

    # Phase alignment removes phase shift
    aligned = metrics.phase_align(x_est, x_true)
    mse_aligned = metrics.mse(aligned, x_true)
    assert mse_aligned < 1e-12

    # PMSE: should be ~0 after alignment
    pmse_val = metrics.pmse(x_est, x_true)
    assert pmse_val < 1e-12

    # PNMSE: normalized variant should be ~0
    pnmse_val = metrics.pnmse(x_est, x_true)
    assert pnmse_val < 1e-12

    # PPSNR: infinite due to perfect alignment
    ppsnr_val = metrics.ppsnr(x_est, x_true, max_val=1.0)
    assert ppsnr_val  > 100
