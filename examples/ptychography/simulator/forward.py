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
    AmplitudeMeasurement(var=noise, label = "meas", damping = 0.3) << fft2(exit_waves)