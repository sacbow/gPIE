"""
Graph Visualization Interface for gPIE

This module provides visualization support for Computational Factor Graphs (CFGs)
using multiple plotting backends and layout algorithms.

Usage:
    >>> from gpie.graph.structure.visualization import visualize_graph
    >>> visualize_graph(graph, backend="bokeh", layout="graphviz", output_path="graph.html")
"""
import networkx as nx
from typing import Literal, Optional

def get_layout_func(layout: Literal["graphviz", "spring", "shell", "kamada_kawai", "circular"]):
    if layout == "graphviz":
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
        except ImportError:
            raise RuntimeError("Graphviz layout requires pygraphviz. Please install it or choose a different layout.")
        return lambda G: graphviz_layout(G, prog="dot")
    elif layout == "spring":
        return nx.spring_layout
    elif layout == "kamada_kawai":
        return nx.kamada_kawai_layout
    elif layout == "shell":
        return nx.shell_layout
    elif layout == "circular":
        return nx.circular_layout
    else:
        raise ValueError(f"Unknown layout: {layout}")


def visualize_graph(
    graph,
    backend: Literal["bokeh", "matplotlib"] = "bokeh",
    layout: Literal["graphviz", "spring", "shell", "kamada_kawai"] = "graphviz",
    output_path: Optional[str] = None,
):
    """
    Visualize a gPIE Graph using the specified backend and layout.

    Args:
        graph (Graph): The graph instance to visualize.
        backend (str): Visualization backend. Supported: "bokeh", "matplotlib".
        layout (str): Layout algorithm. Supported: "graphviz", "spring", "shell", "kamada_kawai".
        output_path (str | None): Output file path (used by some backends, e.g., bokeh HTML).

    Raises:
        ValueError: If an unknown backend or layout is specified.
    """
    if backend == "bokeh":
        try:
            from ._bokeh_vis import render_bokeh_graph
        except ImportError:
            raise ImportError("Bokeh backend not available. Run `pip install bokeh`.")
        return render_bokeh_graph(graph, layout=layout, output_path=output_path)

    elif backend == "matplotlib":
        try:
            from ._matplotlib_vis import render_matplotlib_graph
        except ImportError:
            raise ImportError("Matplotlib backend not available. Run `pip install matplotlib`.")
        return render_matplotlib_graph(graph, layout=layout)

    else:
        raise ValueError(f"Unknown visualization backend: {backend}")
