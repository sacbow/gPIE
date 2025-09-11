from pathlib import Path
from typing import Literal, Optional

import networkx as nx
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, LabelSet, Arrow, NormalHead, CDSView, BooleanFilter
from bokeh.plotting import figure


def render_bokeh_graph(graph, layout: str = "graphviz", output_path: Optional[str] = None):
    """
    Visualize a gPIE Graph using Bokeh.

    Args:
        graph: Graph instance.
        layout: Layout algorithm name.
        output_path: Optional HTML file to save output.

    Raises:
        ValueError: If unknown layout is specified.
    """

    # === 1. Collect nodes and edges ===
    nodes = []
    edges = []

    for node in list(graph._waves) + list(graph._factors):
        nid = id(node)
        label = getattr(node, "label", None) or node.__class__.__name__
        ntype = "wave" if node in graph._waves else "factor"
        nodes.append((nid, {"label": label, "type": ntype}))

        if ntype == "factor":
            for wave in node.inputs.values():
                edges.append((id(wave), nid))
            if node.output:
                edges.append((nid, id(node.output)))

    # === 2. Build graph & layout ===
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    from .visualization import get_layout_func
    layout_func = get_layout_func(layout)
    pos = layout_func(G)

    # === 3. Format node attributes for Bokeh ===
    node_x, node_y, node_type, node_label, node_color = [], [], [], [], []

    for nid, attrs in nodes:
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(-y)
        node_type.append(attrs["type"])
        node_label.append(attrs["label"])
        node_color.append("skyblue" if attrs["type"] == "wave" else "lightgreen")

    source = ColumnDataSource(data=dict(
        x=node_x, y=node_y, type=node_type, label=node_label, color=node_color
    ))

    # === 4. Filter views ===
    is_wave = [t == "wave" for t in node_type]
    is_factor = [t == "factor" for t in node_type]
    wave_view = CDSView(filter=BooleanFilter(is_wave))
    factor_view = CDSView(filter=BooleanFilter(is_factor))

    # === 5. Plot ===
    p = figure(
        title="Computational Factor Graph",
        tools="pan,reset,zoom_in,zoom_out,save",
        width=800, height=600
    )

    p.scatter(x='x', y='y', source=source, size=18,
              marker='circle', color='skyblue', legend_label='Wave', view=wave_view)

    p.scatter(x='x', y='y', source=source, size=18,
              marker='square', color='lightgreen', legend_label='Factor', view=factor_view)

    labels = LabelSet(x="x", y="y", text="label", source=source,
                      text_align="center", text_baseline="bottom",
                      text_font_size="11pt", y_offset=12)
    p.add_layout(labels)

    for src, tgt in edges:
        if src in pos and tgt in pos:
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            p.add_layout(Arrow(end=NormalHead(size=6),
                               x_start=x0, y_start=-y0,
                               x_end=x1, y_end=-y1,
                               line_color="gray", line_width=1.5, line_alpha=0.4))

    p.axis.visible = False
    p.grid.visible = False
    p.legend.visible = False

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file(str(output_path))
        save(p)