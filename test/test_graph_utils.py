import pytest
import numpy as np

from gpie import model, GaussianPrior, fft2, AmplitudeMeasurement
from gpie.core import backend


@model
def simple_graph():
    obj = ~GaussianPrior(event_shape=(4, 4), label="obj", dtype=np.complex64)
    x = fft2(obj)
    AmplitudeMeasurement(var=1e-4, label = "meas") << x


def test_clear_sample_and_summary(capsys):
    backend.set_backend(np)
    g = simple_graph()
    # generate a sample
    g.generate_sample()
    obj = g.get_wave("obj")
    assert obj.get_sample() is not None

    # clear all samples
    g.clear_sample()
    assert obj.get_sample() is None

    # call summary and capture output
    g.summary()
    captured = capsys.readouterr()
    assert "Graph Summary" in captured.out
    assert "Wave nodes" in captured.out
    assert "Factor nodes" in captured.out


def test_to_networkx_structure():
    backend.set_backend(np)
    g = simple_graph()

    G = g.to_networkx()

    # should contain nodes for waves and factors
    wave_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "wave"]
    factor_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "factor"]

    assert len(wave_nodes) >= 1
    assert len(factor_nodes) >= 1

    # edges should exist from wave -> factor and factor -> wave
    edge_labels = [(G.nodes[u]["type"], G.nodes[v]["type"]) for u, v in G.edges()]
    assert ("wave", "factor") in edge_labels or ("factor", "wave") in edge_labels

def test_get_factor_by_label():
    backend.set_backend(np)

    g = simple_graph()

    # Factor をラベルで取得
    meas = g.get_factor("meas")
    from gpie.graph.factor import Factor
    assert isinstance(meas, Factor)
    assert getattr(meas, "label", None) == "meas"

    with pytest.raises(ValueError):
        g.get_factor("not_exist")
