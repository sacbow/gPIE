import importlib.util
import pytest
import numpy as np

from gpie import Graph, GaussianPrior, fft2, PhaseMaskPropagator, AmplitudeMeasurement, pmse
from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_phase_mask

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_coded_diffraction_pattern_inference(xp):
    """Test Coded Diffraction Pattern (CDP) inference with numpy/cupy backends."""
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    # ==== Parameters ====
    H, W = 64, 64  # Reduced size for test speed
    shape = (H, W)
    n_measurements = 4
    noise = 1e-4

    # Generate random phase masks
    phase_masks = [random_phase_mask(shape, rng=rng) for _ in range(n_measurements)]

    # ==== Graph Definition ====
    class CodedDiffractionPattern(Graph):
        def __init__(self, noise, n_measurements, phase_masks):
            super().__init__()
            x = ~GaussianPrior(shape=shape, label="sample")
            for i in range(n_measurements):
                y = PhaseMaskPropagator(phase_masks[i]) @ x
                z = fft2(y)
                with self.observe():
                    _ = AmplitudeMeasurement(var=noise, damping=0.2) @ z
            self.compile()

    g = CodedDiffractionPattern(noise=noise, n_measurements=n_measurements, phase_masks=phase_masks)

    # ==== Initialization and Sampling ====
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=999), update_observed=True)
    X = g.get_wave("sample")
    true_x = X.get_sample()

    # ==== Inference ====
    pse_history = []

    def monitor(graph, t):
        est = graph.get_wave("sample").compute_belief().data
        err = pmse(est, true_x)
        pse_history.append(err)

    g.run(n_iter=100, callback=monitor, verbose=False)

    # ==== Check: PSE decreases significantly ====
    assert pse_history[0] > pse_history[-1], "PSE did not decrease over iterations."
    assert pse_history[-1] < 1e-2, f"Final PSE too high: {pse_history[-1]}"

    # ==== Backend Consistency ====
    # Check that final estimate dtype matches backend
    final_est = g.get_wave("sample").compute_belief().data
    assert isinstance(final_est, xp.ndarray)
    assert final_est.shape == shape


@pytest.mark.parametrize("xp", backend_libs)
def test_coded_diffraction_pattern_and_graph_methods(xp, capsys, tmp_path, monkeypatch):
    """Test CDP inference and cover Graph methods for better coverage."""
    backend.set_backend(xp)
    rng = get_rng(seed=42)

    # ==== Parameters ====
    H, W = 32, 32  # small for test speed
    shape = (H, W)
    n_measurements = 4
    noise = 1e-4
    phase_masks = [random_phase_mask(shape, rng=rng) for _ in range(n_measurements)]

    # ==== Graph Definition ====
    class CodedDiffractionPattern(Graph):
        def __init__(self, noise, n_measurements, phase_masks):
            super().__init__()
            x = ~GaussianPrior(shape=shape, label="sample")
            for pm in phase_masks:
                y = PhaseMaskPropagator(pm) @ x
                z = fft2(y)
                with self.observe():
                    _ = AmplitudeMeasurement(var=noise, damping=0.2) @ z
            self.compile()

    g = CodedDiffractionPattern(noise=noise, n_measurements=n_measurements, phase_masks=phase_masks)

    # ==== Initialization and Sampling ====
    g.set_init_rng(get_rng(seed=1))
    g.generate_sample(rng=get_rng(seed=999), update_observed=True)

    # ==== Summary (stdout captured) ====
    g.summary()
    out, _ = capsys.readouterr()
    assert "Graph Summary" in out

    # ==== get_wave (normal & error cases) ====
    wave = g.get_wave("sample")
    assert wave.label == "sample"
    with pytest.raises(ValueError):
        g.get_wave("nonexistent")

    

   # ==== to_backend switch ====
    if has_cupy:
        backend.set_backend(cp)
        g.to_backend()
        rng = get_rng(seed=123) 
        backend.set_backend(np)
        g.to_backend()
        rng = get_rng(seed=123) 


    # ==== visualize (mock Bokeh save) ====
    monkeypatch.setattr("bokeh.plotting.save", lambda *a, **k: None)
    monkeypatch.setattr("bokeh.plotting.show", lambda *a, **k: None)
    g.visualize()  # should run without error

    # ==== run(verbose=True) ====
    true_x = g.get_wave("sample").get_sample()
    g.run(n_iter=20, verbose=True)  # tqdm path

    # ==== inference with callback ====
    pse_history = []
    def monitor(graph, t):
        est = graph.get_wave("sample").compute_belief().data
        pse_history.append(pmse(est, true_x))

    g.generate_sample(rng=get_rng(seed=999), update_observed=True)
    g.run(n_iter=100, callback=monitor)
    assert pse_history[0] > pse_history[-1]

    # ==== clear_sample ====
    g.clear_sample()
    assert all(w.get_sample() is None for w in g._waves)
