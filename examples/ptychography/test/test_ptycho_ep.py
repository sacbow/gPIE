import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import pytest
from gpie import model, GaussianPrior, AmplitudeMeasurement, fft2, pmse
from gpie.core.backend import np
from gpie.core.rng_utils import get_rng
from examples.ptychography.data.dataset import PtychographyDataset
from examples.ptychography.simulator.scan import generate_raster_positions


# ----------------------------------------------------------------------
# 1. Known-probe ptychography model (AmplitudeMeasurement)
# ----------------------------------------------------------------------
@model
def ptychography_graph_known_probe(
    obj_shape,
    prb,                          # probe is known (numpy/cupy array)
    indices,
    noise: float,
    dtype=np().complex64,
    damping: float = 0.3,
):
    """Ptychography graph with known probe."""
    obj = ~GaussianPrior(event_shape=obj_shape, label="object", dtype=dtype)
    patches = obj.extract_patches(indices)
    exit_waves = prb * patches  # probe acts as constant field in real space
    AmplitudeMeasurement(var=noise, label="meas", damping=damping) << fft2(exit_waves)
    return


# ----------------------------------------------------------------------
# 2. pytest fixture: generate synthetic dataset
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_dataset():
    xp = np()
    rng = get_rng(seed=42)

    ds = PtychographyDataset()
    ds.set_pixel_size(1.0)

    # ---- object: random complex field (128x128) ----
    obj_shape = (128, 128)
    real_part = rng.normal(size=obj_shape)
    imag_part = rng.normal(size=obj_shape)
    obj = (real_part + 1j * imag_part).astype(xp.complex64)
    ds.set_object(obj)

    # ---- probe: circular aperture in *real space* (radius = 16 px) ----
    prb_shape = (64, 64)
    yy, xx = xp.meshgrid(
        xp.arange(prb_shape[0]) - prb_shape[0] / 2,
        xp.arange(prb_shape[1]) - prb_shape[1] / 2,
        indexing="ij",
    )
    r = xp.sqrt(xx**2 + yy**2)
    aperture_radius = 16.0
    prb_amp = xp.where(r <= aperture_radius, 1.0, 0.0).astype(xp.float32)
    prb = prb_amp.astype(xp.complex64)  # purely real amplitude probe
    ds.set_probe(prb)

    # ---- raster scan: 3×3 grid, step = 10.0 µm (9 points total) ----
    scan_gen = generate_raster_positions(stride_um=10.0)
    ds.simulate_diffraction(scan_gen, max_num_points=9, noise=1e-4, rng=rng)

    assert len(ds) == 9, "Expected 9 diffraction patterns."
    return ds


# ----------------------------------------------------------------------
# 3. Test dataset integrity
# ----------------------------------------------------------------------
def test_dataset_integrity(synthetic_dataset):
    ds = synthetic_dataset
    assert ds.obj_shape == (128, 128)
    assert ds.prb_shape == (64, 64)
    assert ds.pixel_size_um == 1.0
    assert ds.size == 9
    for diff in ds._diff_data:
        assert diff.diffraction.shape == ds.prb_shape


# ----------------------------------------------------------------------
# 4. EP reconstruction test (known-probe model)
# ----------------------------------------------------------------------
def test_ep_reconstruction_known_probe(synthetic_dataset):
    xp = np()
    ds = synthetic_dataset
    rng = get_rng(0)

    # ---- extract slice indices from dataset ----
    indices = [d.indices for d in ds._diff_data]
    dataset = xp.stack([d.diffraction for d in ds._diff_data])

    # ---- construct known-probe ptychography model ----
    graph = ptychography_graph_known_probe(
        obj_shape=ds.obj_shape,
        prb=ds.prb,
        indices=indices,
        noise=1e-4,
        dtype=xp.complex64,
        damping=0.3,
    )

    # ---- initialize and run EP iterations ----
    graph.set_init_rng(rng)
    meas = graph.get_factor("meas")
    meas.set_sample(dataset)
    meas.update_observed_from_sample()

    # ---- coverage mask (probe intensity weighted) ----
    coverage_weight = xp.zeros(ds.obj_shape, dtype=xp.float32)

    # probe intensity |prb|^2
    probe_intensity = xp.abs(ds.prb) ** 2

    # accumulate probe intensity across all scan positions
    for d in ds._diff_data:
        if d.indices is not None:
            coverage_weight[d.indices] += probe_intensity

    # normalize and threshold (10% of max)
    max_val = float(coverage_weight.max())
    coverage = coverage_weight > (0.1 * max_val)

    # ---- callback function for monitoring PMSE ----
    def callback(graph, t):
        if t % 10 != 0:  # compute every 10 iterations
            return

        obj_rec = graph.get_wave("object").compute_belief().data[0]
        obj_gt = ds.obj

        masked_rec = obj_rec[coverage]
        masked_gt = obj_gt[coverage]
        err_t = pmse(masked_rec, masked_gt)

        print(f"[Iter {t:03d}] PMSE = {float(err_t):.3e}")
    
    graph.run(n_iter=200, callback = callback)
