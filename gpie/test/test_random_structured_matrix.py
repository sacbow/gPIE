import importlib.util
import pytest
import numpy as np

from gpie import Graph, SupportPrior, fft2, PhaseMaskPropagator, AmplitudeMeasurement, pmse
from gpie.core import backend
from gpie.core.linalg_utils import circular_aperture, random_phase_mask
from gpie.core.rng_utils import get_rng

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


class StructuredRandomModel(Graph):
    """Graph implementing a structured random phase retrieval model."""
    def __init__(self, support, n_layers, phase_masks, var=1e-4, dtype=np.complex128):
        super().__init__()
        x = ~SupportPrior(support=support, label="sample", dtype=dtype)
        for mask in phase_masks:
            x = fft2(mask * x)  # MultiplyConstPropagator applied internally
        with self.observe():
            AmplitudeMeasurement(var=var, damping = 0.2) @ x
        self.compile()


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("obj_dtype", [np.complex128, np.complex64])  # test precision control
def test_structured_random_model_reconstruction(xp, obj_dtype):
    """Test structured random model reconstruction with numpy/cupy and dtype precision control."""
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    H, W = 64, 64
    shape = (H, W)
    support = circular_aperture(shape=shape, radius=0.3)

    # Generate random phase masks
    n_layers = 2
    phase_masks = [random_phase_mask(shape, dtype=xp.complex128, rng=rng) for _ in range(n_layers)]

    # Build graph
    g = StructuredRandomModel(support=support, n_layers=n_layers, phase_masks=phase_masks, var=1e-4, dtype=obj_dtype)

    sample_wave = g.get_wave("sample")
    assert sample_wave.dtype == xp.dtype(obj_dtype)  # dtype should match requested precision

    # Initialize RNG and generate samples
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)
    true_sample = sample_wave.get_sample()

    # Run inference
    def monitor(graph, t):
        est = graph.get_wave("sample").compute_belief().data
        err = pmse(est, true_sample)
        if t % 20 == 0:
            print(f"[t={t}] PMSE = {err:.5e}")

    g.run(n_iter=100, callback=monitor, verbose=False)

    recon = sample_wave.compute_belief().data
    final_err = pmse(recon, true_sample)
    assert final_err < 1e-3
    assert recon.dtype == xp.dtype(obj_dtype)


@pytest.mark.parametrize("xp", backend_libs)
def test_structured_random_model_to_backend(xp):
    """Test graph CPU construction and subsequent transfer to GPU (if cupy available)."""
    backend.set_backend(np)
    rng = get_rng(seed=0)

    H, W = 32, 32
    shape = (H, W)
    support = circular_aperture(shape, radius=0.3)
    phase_masks = [random_phase_mask(shape, dtype=np.complex128, rng=rng) for _ in range(2)]

    g = StructuredRandomModel(support=support, n_layers=2, phase_masks=phase_masks)

    # Transfer to GPU
    if xp is cp:
        backend.set_backend(cp)
        g.to_backend()
        for wave in g._waves:
            assert isinstance(wave.dtype.type, type(cp.complex128().dtype.type))

    # Quick inference to confirm backend consistency
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)
    g.run(n_iter=5, verbose=False)
