import importlib.util
import pytest
import numpy as np

from gpie import Graph, SupportPrior, fft2, AmplitudeMeasurement, mse
from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.core.linalg_utils import circular_aperture, masked_random_array
from gpie.core.rng_utils import get_rng

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


class Holography(Graph):
    """Graph model for holography reconstruction."""
    def __init__(self, var: float, ref_wave, support, dtype = np.complex128):
        super().__init__()
        obj = ~SupportPrior(support=support, label="obj", dtype = dtype)
        with self.observe():
            meas = AmplitudeMeasurement(var=var) @ (fft2(ref_wave + obj))
        self.compile()


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("obj_dtype", [np.complex128, np.complex64])  # test lower precision prior
def test_holography_reconstruction(xp, obj_dtype):
    """Test holography graph with numpy/cupy and varying prior dtype precision."""
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    H, W = 64, 64  # reduced size for test speed
    shape = (H, W)
    noise = 1e-4

    # Prepare support masks and data
    support_x = circular_aperture(shape=shape, radius=0.2, center=(-0.2, -0.2))
    data_x = masked_random_array(support_x, dtype=xp.complex128, rng=rng)
    support_y = circular_aperture(shape=shape, radius=0.2, center=(0.2, 0.2))

    # Build holography graph
    g = Holography(var=noise, ref_wave=data_x, support=support_y, dtype = obj_dtype)

    # Ensure prior wave dtype matches requested obj_dtype (complex64 for low-precision test)
    obj_wave = g.get_wave("obj")
    assert obj_wave.dtype == xp.dtype(obj_dtype)

    # Initialize RNGs and samples
    g.set_init_rng(get_rng(seed=11))
    g.generate_sample(rng=get_rng(seed=9), update_observed=True)

    true_obj = obj_wave.get_sample()

    # Run inference and monitor error decay
    def monitor(graph, t):
        x = graph.get_wave("obj").compute_belief().data
        err = mse(x, true_obj)
        if t % 10 == 0:
            print(f"[t={t}] MSE = {err:.5e}")

    g.run(n_iter=50, callback=monitor, verbose=False)

    # Final reconstruction error should be small
    recon_obj = obj_wave.compute_belief().data
    final_err = mse(recon_obj, true_obj)
    assert final_err < 1e-3

    # Confirm dtype propagation (low precision prior is preserved or lowered)
    assert recon_obj.dtype == xp.dtype(obj_dtype)


@pytest.mark.parametrize("xp", backend_libs)
def test_holography_to_backend(xp):
    """Test holography graph CPU construction then GPU transfer (if cupy available)."""
    backend.set_backend(np)
    rng = get_rng(seed=0)

    H, W = 32, 32
    shape = (H, W)
    support = circular_aperture(shape=shape, radius=0.3)
    ref_wave = masked_random_array(support, dtype=np.complex128, rng=rng)

    # Build on CPU
    g = Holography(var=1e-4, ref_wave=ref_wave, support=support)

    # Transfer graph to GPU if available
    if xp is cp:
        backend.set_backend(cp)
        g.to_backend()
        for wave in g._waves:
            assert wave.dtype == cp.complex128

    # Run a quick inference step (backend-aware execution)
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)
    g.run(n_iter=5, verbose=False)
