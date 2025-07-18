from graph.wave import Wave
from graph.factor import Factor
import numpy as np
import contextlib
import threading

_current_graph = threading.local()


class Graph:
    """
    Graph represents a Computational Factor Graph (CFG) used in gPIE.

    This graph structure encodes the computational dependencies among latent variables (Wave)
    and transformation/measurement operators (Factor), allowing efficient scheduling of 
    message-passing inference such as Belief Propagation.

    Attributes:
        _nodes (set): All Wave and Factor nodes in the graph.
        _waves (set): Subset of _nodes consisting of Wave instances.
        _factors (set): Subset of _nodes consisting of Factor instances.
        _nodes_sorted (list): Topologically sorted list of nodes (forward order).
        _nodes_sorted_reverse (list): Reverse topological order (for backward pass).
        _rng (np.random.Generator): Random number generator used for sampling.
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

    def run(self, n_iter=10, callback=None, rng=None, verbose=True):
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

    def visualize(self, with_labels=True, layout="kamada_kawai", font_size=6, scale=1.0):
        """
        Visualize the graph structure as a directed computational graph.
        Arrows: Wave → Factor → Wave

        Args:
            with_labels (bool): Whether to show text labels on nodes.
            layout (str): Layout type ('spring', 'kamada_kawai', 'shell', etc.)
            font_size (int): Font size for node labels.
            scale (float): Scaling factor for node layout spacing.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        # Add nodes
        for wave in self._waves:
            G.add_node(wave, label=repr(wave), shape='circle')
        for factor in self._factors:
            G.add_node(factor, label=repr(factor), shape='square')

        # Add edges (Wave → Factor, Factor → Wave)
        for wave in self._waves:
            for child in wave.children:
                G.add_edge(wave, child)
        for factor in self._factors:
            if factor.output is not None:
                G.add_edge(factor, factor.output)

        # Layout selection
        if layout == "spring":
            pos = nx.spring_layout(G, seed=42, k=scale)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:  # default to kamada_kawai
            pos = nx.kamada_kawai_layout(G, scale=scale)

        # Draw nodes by shape
        node_shapes = {'circle': [], 'square': []}
        for node in G.nodes:
            shape = G.nodes[node]['shape']
            node_shapes[shape].append(node)

        nx.draw_networkx_nodes(G, pos,
            nodelist=node_shapes['circle'], node_shape='o', node_color='skyblue')
        nx.draw_networkx_nodes(G, pos,
            nodelist=node_shapes['square'], node_shape='s', node_color='lightgreen')

        nx.draw_networkx_edges(G, pos, arrows=True)

        if with_labels:
            labels = {n: G.nodes[n]['label'] for n in G.nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=font_size)

        plt.title("Computational Factor Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()