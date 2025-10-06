"""
Base class for PIE-family ptychographic reconstruction algorithms.

This module defines BasePIE, a unified reconstruction engine compatible with
gPIE's PtychographyDataset data structure and FFT backend abstraction.
It provides a reusable foundation for implementing algorithms such as
PIE, ePIE, and rPIE.
"""

from gpie.core.backend import np
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_normal_array
from gpie.core.fft import get_fft_backend
from gpie.core.types import get_real_dtype
from ..data.dataset import PtychographyDataset


class BasePIE:
    """
    Abstract base class for PIE-type ptychographic reconstruction algorithms.

    This class defines the common workflow and data handling for all PIE-style
    iterative reconstruction methods. Subclasses (PIE, ePIE, rPIE) implement the
    specific object update rule via `_update_object()`.

    Attributes
    ----------
    dataset : PtychographyDataset
        Reference to the static dataset (not modified).
    alpha : float
        Step size for object update.
    obj : ndarray
        Complex-valued reconstruction object.
    prb : ndarray
        Complex-valued probe used for illumination.
    diff_data : list[dict]
        Internal list of diffraction data dicts:
            {"indices", "position", "diffraction", "amplitude", "noise"}.
    fft : gpie.core.fft.FFTBackend
        FFT engine for centered 2D transforms.
    xp : module
        Numerical backend (NumPy or CuPy).
    dtype : np.dtype
        Complex dtype (complex64 or complex128).
    real_dtype : np.dtype
        Corresponding real dtype (float32 or float64).
    callback : callable or None
        Optional callback called as callback(iter, avg_error, obj).
    """

    def __init__(
        self,
        dataset: PtychographyDataset,
        alpha: float = 0.1,
        obj_init=None,
        dtype: str = "complex64",
        callback=None,
        seed: int = None,
    ):
        """Initialize the PIE reconstruction engine."""
        self.xp = np()
        self.dataset = dataset
        self.alpha = self.xp.asarray(alpha)
        self.callback = callback
        self.dtype = getattr(self.xp, dtype)
        self.real_dtype = get_real_dtype(self.dtype)
        self.fft = get_fft_backend()

        # --- Validate dataset ---
        if dataset.obj is None or dataset.prb is None:
            raise ValueError("Dataset must have object and probe set before reconstruction.")
        if len(dataset) == 0:
            raise ValueError("Dataset must contain diffraction data.")

        # --- Local deep copies (preserve dataset immutability) ---
        self.prb = self.xp.array(dataset.prb, dtype=self.dtype)
        self.diff_data = []
        for d in dataset._diff_data:
            diff_copy = self.xp.array(d.diffraction, dtype=get_real_dtype(self.dtype))
            self.diff_data.append(
                {
                    "indices": d.indices,
                    "amplitude": diff_copy
                }
            )

        # --- Initialize object estimate ---
        if obj_init is None:
            rng = get_rng(seed)
            self.obj = random_normal_array(
                shape=dataset.obj.shape,
                dtype=self.dtype,
                rng=rng,
            )
        else:
            self.obj = self.xp.array(obj_init, dtype=self.dtype)


    # -------------------------------------------------------------------------
    # Main optimization loop
    # -------------------------------------------------------------------------
    def run(self, n_iter=100):
        """
        Run the PIE iterative reconstruction process.

        Parameters
        ----------
        n_iter : int
            Number of reconstruction iterations.

        Returns
        -------
        obj : ndarray
            Reconstructed complex object.
        """
        for it in range(n_iter):
            total_error = 0.0

            for d in self.diff_data:
                yy, xx = d["indices"]
                obj_patch = self.obj[yy, xx]
                exit_wave = self.prb * obj_patch

                # Fourier domain projection (amplitude constraint)
                proj_wave, err_val = self._fourier_projector(exit_wave, d["amplitude"])
                total_error += err_val

                # Object update step (implemented in subclass)
                self._update_object(proj_wave, exit_wave, (yy, xx))

            avg_err = total_error / len(self.diff_data)

            if self.callback:
                self.callback(it, avg_err, self.obj)

        return self.obj


    # -------------------------------------------------------------------------
    # Fourier projection (amplitude constraint)
    # -------------------------------------------------------------------------
    def _fourier_projector(self, exit_wave, target_amplitude):
        """
        Project the exit wave onto the measured diffraction amplitude constraint.

        This implementation rescales the complex FFT result directly by
        (target_amp / (|Ψ| + eps)), which avoids costly phase computation via
        exp(angle) and provides better numerical stability.

        Parameters
        ----------
        exit_wave : ndarray (complex)
            Current estimate of the exit wave ψ = P ⊙ O_patch.
        target_amplitude : ndarray (real)
            Measured diffraction amplitude (sqrt of intensity).

        Returns
        -------
        proj_wave : ndarray (complex)
            Projected exit wave in the object domain.
        error : float
            Mean-squared amplitude error between |Ψ| and target amplitude.
        """
        xp = self.xp
        fft_exit = self.fft.fft2_centered(exit_wave)
        current_amp = xp.abs(fft_exit)
        eps = self.real_dtype(1e-8)

        # Stable amplitude correction
        scale = target_amplitude / (current_amp + eps)
        proj_fft = fft_exit * scale
        proj_wave = self.fft.ifft2_centered(proj_fft)

        err = float(xp.mean((current_amp - target_amplitude) ** 2))
        return proj_wave, err


    # -------------------------------------------------------------------------
    # Abstract method: object update rule
    # -------------------------------------------------------------------------
    def _update_object(self, proj_wave, exit_wave, indices):
        """
        Update the object estimate using the projected and current exit waves.

        Must be implemented in subclass (e.g., PIE, ePIE, rPIE).

        Parameters
        ----------
        proj_wave : ndarray
            Projected exit wave after Fourier constraint.
        exit_wave : ndarray
            Current exit wave before projection.
        indices : tuple[slice, slice]
            Object region corresponding to this scan position.
        """
        raise NotImplementedError("Implement _update_object() in subclass.")
