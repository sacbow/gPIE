import numpy as np

def mse(x_est, x_true):
    """
    Mean Squared Error (MSE) between estimated and true arrays.
    Supports complex-valued inputs.
    """
    return np.mean(np.abs(x_est - x_true) ** 2)


def nmse(x_est, x_true):
    """
    Normalized Mean Squared Error (NMSE).
    Scales by norm of the true signal.
    """
    return np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2


def psnr(x_est, x_true, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio (PSNR).
    Assumes signal range [0, max_val]. Typically used for real-valued data.
    """
    mse_val = mse(x_est, x_true)
    if mse_val == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse_val)


def support_error(x_est, x_true, threshold=1e-3):
    """
    Support mismatch between estimated and true sparse signals.
    Useful for sparse recovery problems.
    """
    est_support = np.abs(x_est) > threshold
    true_support = np.abs(x_true) > threshold
    mismatch = np.logical_xor(est_support, true_support)
    return np.sum(mismatch) / len(x_true)

def phase_align(x_est, x_true):
    """
    Align global phase of x_est to match x_true.
    Returns e^{jθ} x_est where θ minimizes MSE to x_true.
    """
    inner_product = np.vdot(x_true, x_est)  # complex conjugate dot
    phase = np.angle(inner_product)
    return x_est * np.exp(-1j * phase)


def pmse(x_est, x_true):
    """
    Phase-aligned Mean Squared Error (PMSE).
    Removes global phase before computing MSE.
    """
    aligned = phase_align(x_est, x_true)
    return mse(aligned, x_true)


def pnmse(x_est, x_true):
    """
    Phase-aligned Normalized MSE.
    """
    aligned = phase_align(x_est, x_true)
    return nmse(aligned, x_true)


def ppsnr(x_est, x_true, max_val=1.0):
    """
    Phase-aligned PSNR.
    """
    aligned = phase_align(x_est, x_true)
    return psnr(aligned, x_true, max_val=max_val)
