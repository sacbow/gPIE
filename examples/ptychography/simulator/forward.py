from gpie import Graph, model, GaussianPrior, fft2, AmplitudeMeasurement, replicate
from typing import Tuple, List

# ---------- model definition ----------

@model
def ptychography_graph(
    obj_shape: Tuple[int, int],
    prb_shape: Tuple[int, int],
    indices: List[Tuple[slice, slice]],
    noise: float,
) -> Graph:
    """
    Construct a factor graph for ptychography simulation.

    Parameters
    ----------
    obj_shape : (H, W)
        Shape of the complex object.
    prb_shape : (h, w)
        Shape of the probe.
    indices : list of slices
        Slice indices to extract patches from the object.
    noise : float
        Gaussian noise variance (σ²).

    Returns
    -------
    Graph : gpie.graph.Graph
        Assembled factor graph ready for sampling or inference.
    """
    # Object and probe priors
    obj = ~GaussianPrior(event_shape = obj_shape, label="object", dtype="complex64")
    prb = ~GaussianPrior(event_shape = prb_shape, label="probe", dtype="complex64")

    # Extract patches from object
    patches = obj.extract_patches(indices)

    # Replicate probe across batch
    prb_repl = replicate(prb, batch_size=len(indices))

    # Multiply exit waves
    exit_waves = patches * prb_repl

    # Amplitude measurement (i.e., diffraction)
    AmplitudeMeasurement(var=noise, label = "meas") << fft2(exit_waves)



# ---------- simulating ptychography ----------

from typing import List, Tuple
from ..data.dataset import PtychographyDataset
from ..data.diffraction_data import DiffractionData
from ..utils.geometry import slices_from_positions
from .probe import generate_probe
from .scan import generate_raster_positions, generate_fermat_spiral_positions

from gpie import Graph
from gpie.core.backend import np


def generate_diffraction_data(
    dataset: PtychographyDataset,
    indices: List[Tuple[slice, slice]],
    positions_real: List[Tuple[float, float]],
    noise: float = 1e-4,
    rng: "np.random.Generator" = None,
) -> List[DiffractionData]:
    """
    Generate synthetic diffraction data given object, probe, and precomputed indices.

    Parameters
    ----------
    dataset : PtychographyDataset
        The container holding object, probe, and metadata.
    indices : list of tuple[slice, slice]
        Slice objects corresponding to patches in the object array.
    positions_real : list of tuple[float, float]
        Real-space scan positions (μm) corresponding to each index.
    noise : float
        Variance of Gaussian noise to be added in amplitude domain.
    rng : np.random.Generator
        Optional random number generator.

    Returns
    -------
    List[DiffractionData]
        Simulated diffraction patterns at each real-space position.
    """
    if dataset.obj is None or dataset.prb is None:
        raise ValueError("Object and probe must be set before simulation.")

    prb_shape = dataset.prb.shape
    obj_shape = dataset.obj.shape

    if len(indices) != len(positions_real):
        raise ValueError("Length of indices and positions_real must match.")

    # Build the factor graph
    g = ptychography_graph(
        obj_shape=obj_shape,
        prb_shape=prb_shape,
        indices=indices,
        noise=noise,
    )

    # Inject samples
    g.get_wave("object").set_sample(dataset.obj)
    g.get_wave("probe").set_sample(dataset.prb)

    # Simulate noisy measurements
    g.generate_sample(rng=rng, update_observed=True)

    # Extract generated diffraction patterns
    meas = g.get_factor("meas")
    diffs = meas.get_sample()

    # Wrap into DiffractionData
    results = []
    for i, (pos_real, idx) in enumerate(zip(positions_real, indices)):
        diff = diffs[i]
        results.append(
            DiffractionData(
                position=pos_real,      
                diffraction=diff,
                noise=noise,
                indices=idx
            )
        )

    return results