import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional
from pathlib import Path


def _matplotlib_vis(graph, layout: str = "spring", output_path: Optional[str] = None):
    """
    Visualize the factor graph using matplotlib.

    Args:
        graph (Graph): The Computational Factor Graph instance.
        layout (str): Layout algorithm to use ("spring", "shell", "kamada_kawai", etc.).
        output_path (str or None): If given, save to this path as PNG. Else, display with plt.show().
    """
    G = nx.DiGraph()
    node_attrs = {}

    # Collect nodes and edges
    for node in list(graph._waves) + list(graph._factors):
        nid = id(node)
        label = getattr(node, "label", None) or node.__class__.__name__
        ntype = "wave" if node in graph._waves else "factor"
        G.add_node(nid)
        node_attrs[nid] = {"label": label, "type": ntype, "ref": node}

        if ntype == "factor":
            for wave in node.inputs.values():
                G.add_edge(id(wave), nid)
            if node.output:
                G.add_edge(nid, id(node.output))

    # Positioning
    from .visualization import get_layout_func
    pos_func = get_layout_func(layout)
    pos = pos_func(G)

    # Categorize nodes
    wave_nodes = [nid for nid, attr in node_attrs.items() if attr["type"] == "wave"]
    factor_nodes = [nid for nid, attr in node_attrs.items() if attr["type"] == "factor"]

    # Draw
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=wave_nodes, node_color="skyblue", node_shape="o", label="Wave")
    nx.draw_networkx_nodes(G, pos, nodelist=factor_nodes, node_color="lightgreen", node_shape="s", label="Factor")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", edge_color="gray", alpha=0.5)
    labels = {nid: attr["label"] for nid, attr in node_attrs.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.axis("off")
    plt.legend()
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), bbox_inches="tight")
        plt.close()
    else:
        plt.show()
