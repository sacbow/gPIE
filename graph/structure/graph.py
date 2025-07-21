from ..wave import Wave
from ..factor import Factor
import numpy as np
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

        self._rng = np.random.default_rng()  # default RNG for sampling
    
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
            rng (np.random.Generator or None): RNG used to sample latent variables.
            update_observed (bool): If True, automatically update observed data
                                for all Measurement nodes based on the sampled latent state.
        """

        if rng is None:
            rng = self._rng

        for node in self._nodes_sorted:
            if hasattr(node, "generate_sample"):
                node.generate_sample(rng)

        if update_observed:
            for factor in self._factors:
                if hasattr(factor, "update_observed_from_sample"):
                    factor.update_observed_from_sample()


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
    
    def visualize(self):
        from bokeh.plotting import figure, show
        from bokeh.models import ColumnDataSource, LabelSet, Arrow, NormalHead
        from bokeh.io import output_notebook
        import numpy as np

        output_notebook()

        def list_of_dicts_to_dict_of_lists(ld):
            return {key: [d[key] for d in ld] for key in ld[0]}

        nodes = []
        edges = []

        for i, node in enumerate(list(self._waves) + list(self._factors)):
            gen = getattr(node, "generation", 0)
            label = getattr(node, "label", None)
            if label is None:
                label = node.__class__.__name__

            node_type = "wave" if node in self._waves else "factor"

            group = [n for n in (self._waves if node_type == "wave" else self._factors) if getattr(n, "generation", 0) == gen]
            index_in_group = group.index(node)
            y_offset = index_in_group * 0.4 - (len(group) - 1) * 0.2  

            nodes.append(dict(
                id=id(node),
                label=label,
                type=node_type,
                x=gen * 1.5,
                y=y_offset,
                generation=gen
            ))

            if node_type == "factor":
                for wave in node.inputs.values():
                    edges.append((id(wave), id(node)))
                if node.output:
                    edges.append((id(node), id(node.output)))

        wave_data = list_of_dicts_to_dict_of_lists([n for n in nodes if n["type"] == "wave"])
        factor_data = list_of_dicts_to_dict_of_lists([n for n in nodes if n["type"] == "factor"])
        all_data = list_of_dicts_to_dict_of_lists(nodes)

        wave_source = ColumnDataSource(wave_data)
        factor_source = ColumnDataSource(factor_data)
        label_source = ColumnDataSource(all_data)

        xs = [n["x"] for n in nodes]
        ys = [n["y"] for n in nodes]

        x_margin = 1.0
        y_margin = 1.0

        x_min = min(xs) - x_margin
        x_max = max(xs) + x_margin
        y_min = min(ys) - y_margin
        y_max = max(ys) + y_margin

        p = figure(
            title="Computational Factor Graph",
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            tools="pan,zoom_in,zoom_out,reset,save",
            width=800,
            height=600,
        )


        p.scatter(x="x", y="y", source=wave_source, size=20, color="skyblue", marker="circle", legend_label="Wave")
        p.scatter(x="x", y="y", source=factor_source, size=20, color="lightgreen", marker="square", legend_label="Factor")

        labels = LabelSet(
            x="x",
            y="y",
            text="label",
            source=label_source,
            text_align="center",
            text_baseline="bottom",  
            text_font_size="9pt",
            y_offset=10,             
        )

        p.add_layout(labels)
        gen_lookup = {n["id"]: n.get("generation", 0) for n in nodes}
        pos_lookup = {n["id"]: (n["x"], n["y"]) for n in nodes}

        for src_id, tgt_id in edges:
            src_gen = gen_lookup.get(src_id, 0)
            tgt_gen = gen_lookup.get(tgt_id, 0)

            if src_gen <= tgt_gen:
                start_id, end_id = src_id, tgt_id
            else:
                start_id, end_id = tgt_id, src_id

            x0, y0 = pos_lookup[start_id]
            x1, y1 = pos_lookup[end_id]

            arrow = Arrow(end=NormalHead(size=8),
                        x_start=x0, y_start=y0,
                        x_end=x1, y_end=y1,
                        line_width=1.5, line_color="gray")

            p.add_layout(arrow)

        p.axis.visible = False
        p.grid.visible = False
        show(p)

