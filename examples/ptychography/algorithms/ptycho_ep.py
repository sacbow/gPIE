"""
EP-based ptychographic reconstruction algorithm using the gPIE graphical model engine.

This implementation constructs a ptychographic factor graph via `ptychography_graph`
and runs Expectation Propagation (EP) inference to reconstruct the object and probe
from measured diffraction amplitudes.

Workflow:
    1. Extract object/probe shape, indices, and noise level from the dataset.
    2. Build the factor graph with `ptychography_graph`.
    3. Inject observed amplitude data into the measurement factor.
    4. Initialize RNG and run EP iterations via `graph.run()`.
"""

from gpie import  model, GaussianPrior, AmplitudeMeasurement, replicate, fft2, pmse
from gpie.core.backend import np
from gpie.core.rng_utils import get_rng
from gpie.core.types import get_real_dtype
from ..data.dataset import PtychographyDataset
from typing import Tuple, List, Optional

@model
def ptychography_graph(
    obj_shape: Tuple[int, int],
    prb_shape: Tuple[int, int],
    indices: List[Tuple[slice, slice]],
    noise: float,
    dtype = np().complex64,
    damping :float = 0.0 ):
    # Object and probe priors
    obj = ~GaussianPrior(event_shape = obj_shape, label="object", dtype=dtype)
    prb = ~GaussianPrior(event_shape = prb_shape, label="probe", dtype=dtype)
    patches = obj.extract_patches(indices)
    prb_repl = replicate(prb, batch_size=len(indices))
    exit_waves = patches * prb_repl
    AmplitudeMeasurement(var=noise, label = "meas", damping = damping) << fft2(exit_waves)
    return



class PtychoEP:
    """
    Expectation Propagation (EP) reconstruction engine for ptychography.

    This class wraps the gPIE factor graph model to perform EP-based
    message passing between object, probe, and measurement factors.

    Parameters
    ----------
    dataset : PtychographyDataset
        Dataset containing object, probe, scan positions, and diffraction amplitudes.
    noise : float
        Measurement noise variance (σ²).
    damping : float, optional
        Damping factor for EP message updates (0 ≤ damping ≤ 1).
    dtype : str, optional
        Data type for computation ("complex64" or "complex128").
    seed : int, optional
        Random seed for reproducible initialization.
    """

    def __init__(
        self,
        dataset: PtychographyDataset,
        noise: float = 1e-6,
        damping: float = 0.0,
        dtype: str = "complex64",
        seed: int = 0,
    ):
        # --- backend setup ---
        self.xp = np()
        self.dataset = dataset
        self.dtype = getattr(self.xp, dtype)
        self.real_dtype = get_real_dtype(self.dtype)
        self.noise = noise
        self.damping = damping
        self.rng = get_rng(seed)

        # --- dataset validation ---
        if dataset.obj is None or dataset.prb is None:
            raise ValueError("Dataset must have object and probe set before EP reconstruction.")
        if len(dataset) == 0:
            raise ValueError("Dataset must contain diffraction data before EP reconstruction.")

        # --- extract model parameters ---
        self.obj_shape = dataset.obj.shape
        self.prb_shape = dataset.prb.shape
        self.indices = [d.indices for d in dataset._diff_data]

        # --- build factor graph ---
        self.graph = ptychography_graph(
            obj_shape=self.obj_shape,
            prb_shape=self.prb_shape,
            indices=self.indices,
            noise=self.noise,
            dtype=self.dtype,
            damping=self.damping,
        )

        # --- assign RNG for message initialization ---
        self.graph.set_init_rng(self.rng)

        # --- inject observed data (measured amplitudes) ---
        meas = self.graph.get_factor("meas")

        # stack amplitude patterns into a single array
        amplitudes = self.xp.stack([self.xp.abs(d.diffraction) for d in dataset._diff_data]).astype(
            self.real_dtype
        )

        meas.set_sample(amplitudes)
        meas.update_observed_from_sample()

    # -------------------------------------------------------------------------
    # main EP loop
    # -------------------------------------------------------------------------
    def run(self, n_iter=50, callback=None, verbose=True):
        """
        Run EP-based reconstruction via message passing on the factor graph.

        Parameters
        ----------
        n_iter : int
            Number of EP iterations.
        callback : callable, optional
            Function called each iteration as callback(graph, t).
        verbose : bool
            Show progress bar via tqdm (if available).

        Returns
        -------
        graph : gpie.Graph
            The factor graph after EP convergence.
        """
        self.graph.run(n_iter=n_iter, callback=callback, verbose=verbose)

    # -------------------------------------------------------------------------
    # convenience accessors
    # -------------------------------------------------------------------------
    def get_object(self):
        """Return the current reconstructed object sample."""
        return self.graph.get_wave("object").compute_belief().data[0]

    def get_probe(self):
        """Return the current reconstructed probe sample."""
        return self.graph.get_wave("probe").compute_belief().data[0]
