import importlib.util
import pytest
import numpy as np

from gpie import Graph, SupportPrior, ConstWave, fft2, AmplitudeMeasurement, mse
from gpie.core import backend
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
    """Holography graph with uncertain reference wave (ConstWave)."""
    def __init__(self, var: float, ref, support, large_value=1e3, dtype=np.complex64):
        super().__init__()
        obj = ~SupportPrior(support=support, label="obj", dtype=dtype)
        ref_wave = ~ConstWave(data=ref, large_value=large_value, label="ref")
        with self.observe():
            AmplitudeMeasurement(var=var) @ (fft2(ref_wave + obj))
        self.compile()


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("large_value", [1e3, 1e4])  # high precision vs uncertain ref wave
def test_holography_with_uncertain_ref(xp, large_value):
    """Test holography reconstruction with uncertain reference wave (ConstWave)."""
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    H, W = 64, 64
    shape = (H, W)
    noise = 1e-4

    # Supports and masked arrays
    support_ref = circular_aperture(shape=shape, radius=0.2, center=(-0.2, -0.2))
    ref = masked_random_array(support_ref, dtype=xp.complex64, rng=rng)
    support_obj = circular_aperture(shape=shape, radius=0.2, center=(0.2, 0.2))

    # Build holography graph
    g = Holography(var=noise, ref=ref, support=support_obj, large_value=large_value, dtype=xp.complex64)

    # Verify waves and dtypes
    obj_wave = g.get_wave("obj")
    ref_wave = g.get_wave("ref")
    assert obj_wave.dtype == xp.complex64
    assert ref_wave.dtype == xp.complex64

    # Initialize and generate samples
    g.set_init_rng(get_rng(seed=11))
    g.generate_sample(rng=get_rng(seed=9), update_observed=True)
    true_obj = obj_wave.get_sample()

    # Run inference and monitor error decrease
    errors = []

    def monitor(graph, t):
        est = graph.get_wave("obj").compute_belief().data
        err = mse(est, true_obj)
        errors.append(err)
        if t % 10 == 0:
            print(f"[t={t}] MSE = {err:.5e}")

    g.run(n_iter=50, callback=monitor, verbose=False)

    # Ensure final reconstruction is sufficiently accurate
    final_err = errors[-1]
    assert final_err < 1e-2

    # Verify error decreases monotonically in early iterations (robustness check)
    assert errors[0] > errors[min(5, len(errors) - 1)]


@pytest.mark.parametrize("xp", backend_libs)
def test_holography_with_uncertain_ref_to_backend(xp):
    """Test backend transfer of holography graph with uncertain ref wave."""
    backend.set_backend(np)
    rng = get_rng(seed=0)

    H, W = 32, 32
    shape = (H, W)
    support_ref = circular_aperture(shape, radius=0.2, center=(-0.2, -0.2))
    ref = masked_random_array(support_ref, dtype=np.complex64, rng=rng)
    support_obj = circular_aperture(shape, radius=0.2, center=(0.2, 0.2))

    g = Holography(var=1e-4, ref=ref, support=support_obj, large_value=1e4)

    if xp is cp:
        backend.set_backend(cp)
        g.to_backend()
        for wave in g._waves:
            assert isinstance(wave.dtype.type, type(cp.complex64().dtype.type))

    # Quick run to ensure backend consistency
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=2), update_observed=True)
    g.run(n_iter=5, verbose=False)
