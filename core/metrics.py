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
