import importlib.util
import tempfile
import pytest
import numpy as np
import os

from gpie import model, observe
from gpie import SparsePrior, GaussianMeasurement, UnitaryPropagator
from gpie.graph.structure.visualization import visualize_graph
from gpie.core.linalg_utils import random_unitary_matrix

# Optional dependencies
bokeh_spec = importlib.util.find_spec("bokeh")
matplotlib_spec = importlib.util.find_spec("matplotlib")
pygraphviz_spec = importlib.util.find_spec("pygraphviz")

has_bokeh = bokeh_spec is not None
has_matplotlib = matplotlib_spec is not None
has_graphviz = pygraphviz_spec is not None


# --- Define test graph ---
@pytest.fixture(scope="module")
def test_graph():
    @model
    def cs_model():
        x = ~SparsePrior(rho=0.1, event_shape=(32,), damping=0.03, label="x", dtype=np.complex64)
        U = random_unitary_matrix(32, dtype=np.complex64)
        with observe():
            GaussianMeasurement(var=1e-3) << (UnitaryPropagator(U) @ x)
    return cs_model()


# --- Backends and Layouts ---
backends = []
if has_bokeh:
    backends.append("bokeh")
if has_matplotlib:
    backends.append("matplotlib")

layouts = ["graphviz", "spring", "shell", "kamada_kawai", "circular"]


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("layout", layouts)
def test_visualization_runs_without_error(test_graph, backend, layout):
    """
    Check that the visualization function runs without throwing errors
    for all supported backends and layout combinations.

    Uses a closed temporary file to avoid write permission errors on Windows.
    """
    suffix = ".html" if backend == "bokeh" else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        visualize_graph(test_graph, backend=backend, layout=layout, output_path=tmp_path)
    except RuntimeError as e:
        if layout == "graphviz" and not has_graphviz:
            pytest.skip("Graphviz (pygraphviz) not available")
        else:
            raise
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
