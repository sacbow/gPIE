from ..wave import Wave
from ..factor import Factor
from ...core.rng_utils import get_rng
import contextlib
import threading

_current_graph = threading.local()


class Graph:
    """
    The central coordinator of a Computational Factor Graph (CFG) in gPIE.

    A Graph manages the connectivity, compilation, scheduling, and execution
    of all `Wave` (latent variables) and `Factor` (probabilistic operators) nodes.

    Responsibilities:
    - Collect Wave and Factor instances into a coherent model
    - Compile the model into a topologically sorted DAG
    - Propagate precision modes forward/backward for consistency
    - Perform forward-backward inference via message passing
    - Support sampling and observation via Measurement nodes

    Key Concepts:
    - Compilation is required before inference; it discovers all reachable nodes.
    - Message passing is done in topological order (`forward`) and reverse (`backward`).
    - The graph supports sampling (`generate_sample`) and visualization (`visualize`).

    Usage Example:
        >>> g = Graph()
        >>> with g.observe():
        >>>     z = GaussianMeasurement(...) @ (x + y)
        >>> g.compile()
        >>> g.run(n_iter=10)

    Internal State:
        _nodes (set): All Wave and Factor nodes in the graph
        _waves (set): Subset of Wave instances
        _factors (set): Subset of Factor instances
        _nodes_sorted (list): Nodes in forward topological order
        _nodes_sorted_reverse (list): Nodes in reverse topological order
        _rng (np.random.Generator): Default RNG for sampling and initialization
    """

    def __init__(self):
        self._nodes = set()
        self._waves = set()
        self._factors = set()
        self._nodes_sorted = None
        self._nodes_sorted_reverse = None

        self._rng = get_rng()  # default RNG for sampling
    
    @contextlib.contextmanager
    def observe(self):
        """
        Context manager for automatic measurement registration.
        Usage:
            with self.observe():
                Z = AmplitudeMeasurement(...) @ x
        """
        _current_graph.value = self
        try:
            yield
        finally:
            _current_graph.value = None
    
    @staticmethod
    def get_active_graph():
        """Return the current graph context if inside a `with observe()` block."""
        return getattr(_current_graph, "value", None)


    def compile(self):
        """
        Discover the full computational factor graph topology starting from Measurement nodes.

        This method performs the following steps:
            1. Detects Measurement objects defined on the Graph.
            2. Traverses the graph in reverse from Measurements to Priors,
            registering all Waves and Factors.
            3. Sorts all nodes topologically based on generation index.
            4. Propagates precision mode (scalar/array) forward and backward.
            5. Assigns default precision mode where unresolved.
            6. Finalizes wave structure (e.g., shape/dtype assertions).

        Raises:
            ValueError: If graph contains invalid or disconnected components.
        """

        self._nodes.clear()
        self._waves.clear()
        self._factors.clear()

        # --- Step 1: Initialize unseen set with Measurement objects ---
        unseen = set()
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, Factor) and obj.output is None:
                unseen.add(obj)

        # --- Step 2: Recursively traverse the graph in reverse ---
        while unseen:
            factor = unseen.pop()
            if factor in self._factors:
                continue  # already processed

            self._factors.add(factor)
            self._nodes.add(factor)

            for wave in factor.inputs.values():
                if wave not in self._waves:
                    self._waves.add(wave)
                    self._nodes.add(wave)

                parent = wave.parent
                if parent is not None and parent not in self._factors:
                    unseen.add(parent)

        # --- Step 3: Topological sort by generation ---
        self._nodes_sorted = sorted(self._nodes, key=lambda x: x.generation)
        self._nodes_sorted_reverse = list(reversed(self._nodes_sorted))

        # --- Step 4: Precision mode propagation ---
        for node in self._nodes_sorted:
            if hasattr(node, "set_precision_mode_forward"):
                node.set_precision_mode_forward()

        for node in self._nodes_sorted_reverse:
            if hasattr(node, "set_precision_mode_backward"):
                node.set_precision_mode_backward()

        # --- Step 5: Default precision mode fallback ---
        for wave in self._waves:
            if wave.precision_mode is None:
                wave._set_precision_mode("scalar")

        for factor in self._factors:
            if factor.precision_mode is None:
                factor._set_precision_mode("scalar")

        # --- Step 6: Finalize wave structure ---
        for wave in self._waves:
            if hasattr(wave, "finalize_structure"):
                wave.finalize_structure()
    
    def to_backend(self) -> None:
        """
        Move all graph data (Waves, Factors, and their internal arrays) to the current backend (NumPy or CuPy),
        explicitly converting samples and observed values to avoid implicit conversion errors.
        """
        from ...core.backend import np
        from ...core.rng_utils import get_rng
        import importlib

        has_cupy = importlib.util.find_spec("cupy") is not None
        if has_cupy:
            import cupy as cp

        # Waves: belief, samples, etc.
        for wave in self._waves:
            if hasattr(wave, "to_backend"):
                wave.to_backend()

            # Explicitly handle wave._sample conversion
            if hasattr(wave, "_sample") and wave._sample is not None:
                if has_cupy and isinstance(wave._sample, (cp.ndarray,)):
                    if np().__name__ == "numpy":  # CuPy→NumPy
                        wave._sample = wave._sample.get()
                    else:  # CuPy→CuPy (dtype sync)
                        wave._sample = cp.asarray(wave._sample, dtype=wave.dtype)
                else:
                    if np().__name__ == "cupy":  # NumPy→CuPy
                        wave._sample = cp.asarray(wave._sample)
                    else:  # NumPy→NumPy (dtype sync)
                        wave._sample = np().asarray(wave._sample, dtype=wave.dtype)

        # Factors: Prior, Measurement, Propagators, etc.
        for factor in self._factors:
            if hasattr(factor, "to_backend"):
                factor.to_backend()

            # Measurement-specific: observed/sample arrays
            if hasattr(factor, "observed") and factor.observed is not None:
                factor.observed.to_backend()

            if hasattr(factor, "_sample") and factor._sample is not None:
                if has_cupy and isinstance(factor._sample, (cp.ndarray,)):
                    if np().__name__ == "numpy":
                        factor._sample = factor._sample.get()
                    else:
                        factor._sample = cp.asarray(factor._sample, dtype=factor.expected_observed_dtype)
                else:
                    if np().__name__ == "cupy":
                        factor._sample = cp.asarray(factor._sample)
                    else:
                        factor._sample = np().asarray(factor._sample, dtype=factor.expected_observed_dtype)

        # RNG sync
        self._rng = get_rng()
        for factor in self._factors:
            if hasattr(factor, "_init_rng"):
                factor._init_rng = get_rng()
        for wave in self._waves:
            if hasattr(wave, "_init_rng"):
                wave._init_rng = get_rng()


    
    def get_wave(self, label: str):
        """
        Retrieve the Wave instance with the given label.

        Args:
            label (str): Label assigned to the Wave.

        Returns:
            Wave instance with the specified label.

        Raises:
            ValueError: If no wave with the given label exists or if multiple waves share the label.
        """
        matches = [w for w in self._waves if getattr(w, "label", None) == label]
        if not matches:
            raise ValueError(f"No wave found with label '{label}'")
        if len(matches) > 1:
            raise ValueError(f"Multiple waves found with label '{label}'")
        return matches[0]


    def forward(self):
        """Execute forward message passing in cached generation order."""
        for node in self._nodes_sorted:
            node.forward()

    def backward(self):
        """Execute backward message passing in reverse cached order."""
        for node in self._nodes_sorted_reverse:
            node.backward()

    def run(self, n_iter=10, callback=None, rng=None, verbose=False):
        """
        Run multiple rounds of belief propagation with optional progress bar.

        Args:
            n_iter (int): Number of forward-backward iterations.
            callback (callable or None): Function called as callback(graph, t).
            rng (np.random.Generator or None): Optional RNG.
            verbose (bool): Whether to show progress bar (requires tqdm).
        """
        rng = rng or self._rng

        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n_iter), desc="BP Iteration")
            except ImportError:
                print("[Graph.run] tqdm not found. Running without progress bar.")
                iterator = range(n_iter)
        else:
            iterator = range(n_iter)

        for t in iterator:
            self.forward()
            self.backward()
            if callback is not None:
                callback(self, t)


    def generate_sample(self, rng=None, update_observed: bool = True):
        """
        Generate a full sample from the generative model defined by the graph.
        
        Args:
            rng: RNG used for latent and observed sampling (optional).
            update_observed: If True, observed data is updated from the sample.
        """
        rng = rng or get_rng()  # ← backend-aware RNG utility

        # 1. Generate latent samples from priors/propagators
        for node in self._nodes_sorted:
            if isinstance(node, Wave):
                node._generate_sample(rng=rng)

        # 2. Generate noisy observed samples at Measurement nodes
        for meas in self._factors:
            if hasattr(meas, "_generate_sample") and callable(meas._generate_sample):
                meas._generate_sample(rng)

        # 3. Promote synthetic observed samples to actual observations
        if update_observed:
            for meas in self._factors:
                if hasattr(meas, "update_observed_from_sample") and callable(meas.update_observed_from_sample):
                    meas.update_observed_from_sample()




    def set_init_rng(self, rng):
        """
        Propagate RNG to all factors that support message initialization.
        This is separate from sample RNG, and is used for initial message setup.

        Args:
            rng (np.random.Generator): RNG to be used for initializing messages.
        """
        for factor in self._factors:
            if hasattr(factor, "set_init_rng"):
                factor.set_init_rng(rng)

        for wave in self._waves:
            if hasattr(wave, "set_init_rng"):
                wave.set_init_rng(rng)

    def clear_sample(self):
        """
        Clear all sample values stored in Wave nodes in the graph.
        """
        for wave in self._waves:
            wave.clear_sample()

    def summary(self):
        """Print a summary of the graph structure."""
        print("Graph Summary:")
        print(f"- {len(self._waves)} Wave nodes")
        print(f"- {len(self._factors)} Factor nodes")
    

    def visualize(self, output_path="graph.html"):
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        from bokeh.plotting import figure, show, save, output_file
        from bokeh.models import ColumnDataSource, LabelSet, Arrow, NormalHead, CDSView, BooleanFilter
        from bokeh.io import output_notebook
        from bokeh.io.export import export_png
        import os

        output_notebook()

        # === 1. ノードとエッジの収集 ===
        nodes = []
        edges = []

        for node in list(self._waves) + list(self._factors):
            nid = id(node)
            label = getattr(node, "label", None) or node.__class__.__name__
            ntype = "wave" if node in self._waves else "factor"
            nodes.append((nid, {"label": label, "type": ntype}))

            if ntype == "factor":
                for wave in node.inputs.values():
                    edges.append((id(wave), nid))
                if node.output:
                    edges.append((nid, id(node.output)))

        # === 2. グラフ構築 & レイアウト ===
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        pos = graphviz_layout(G, prog="dot")

        # === 3. ノード属性をBokeh用に整形 ===
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

        # === 4. View 定義 ===
        is_wave = [t == "wave" for t in node_type]
        is_factor = [t == "factor" for t in node_type]
        wave_view = CDSView(filter=BooleanFilter(is_wave))
        factor_view = CDSView(filter=BooleanFilter(is_factor))

        # === 5. プロット作成 ===
        p = figure(title="Computational Factor Graph",
               tools="pan,reset,zoom_in,zoom_out,save",
               width=800, height=600)

        p.scatter(x='x', y='y', source=source, size=18,
              marker='circle', color='skyblue', legend_label='Wave', view=wave_view)

        p.scatter(x='x', y='y', source=source, size=18,
              marker='square', color='lightgreen', legend_label='Factor', view=factor_view)

        labels = LabelSet(x="x", y="y", text="label", source=source,
                      text_align="center", text_baseline="bottom",
                      text_font_size="11pt", y_offset=12)
        p.add_layout(labels)

        # エッジ
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

        # === 6. 画像ファイルに保存 ===
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_file(str(output_path))
        save(p)