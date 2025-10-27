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

def test_graph_set_init_strategy_manual_and_sample(monkeypatch):
    """Graph.set_init_strategy should configure Prior init modes via label."""
    backend.set_backend(np)
    g = simple_graph()

    # --- manual init with ndarray ---
    init_data = np.ones((1, 4, 4), dtype=np.complex64)
    g.set_init_strategy(label="obj", mode="manual", data=init_data, verbose=False)

    prior = g.get_wave("obj").parent
    assert np.allclose(prior._manual_init_msg.data, 1.0)
    assert prior._init_strategy == "manual"

    # --- change to sample mode ---
    g.set_init_strategy(label="obj", mode="sample", verbose=False)
    assert prior._init_strategy == "sample"

    # --- change to uninformative mode ---
    g.set_init_strategy(label="obj", mode="uninformative", verbose=False)
    assert prior._init_strategy == "uninformative"


def test_graph_set_init_strategy_invalid_cases():
    """Graph.set_init_strategy should raise clear errors for invalid usage."""
    backend.set_backend(np)
    g = simple_graph()

    # manual mode but missing data
    with pytest.raises(ValueError, match="missing"):
        g.set_init_strategy(label="obj", mode="manual", verbose=False)

    # invalid mode name
    with pytest.raises(ValueError, match="Invalid init strategy"):
        g.set_init_strategy(label="obj", mode="nonsense", verbose=False)


def test_graph_set_all_init_strategies_error_cases():
    """Graph.set_all_init_strategies should raise for malformed entries."""
    backend.set_backend(np)
    g = simple_graph()

    # manual mode but missing data
    bad_dict = {"obj": ("manual", None)}
    with pytest.raises(ValueError, match="requires ndarray"):
        g.set_all_init_strategies(bad_dict, verbose=False)

    # invalid mode string (handled inside Prior.set_init_strategy)
    bad_dict = {"obj": ("invalid_mode", None)}
    with pytest.raises(ValueError, match="Invalid init strategy"):
        g.set_all_init_strategies(bad_dict, verbose=False)
