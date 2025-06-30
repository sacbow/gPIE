from graph.wave import Wave
from graph.factor import Factor
import numpy as np


class Graph:
    def __init__(self):
        self._nodes = set()
        self._waves = set()
        self._factors = set()
        self._nodes_sorted = None
        self._nodes_sorted_reverse = None

        self._rng = np.random.default_rng()  # default RNG for sampling

    def register_wave(self, wave):
        """Register a Wave node and its parent Factor (if present)."""
        if wave not in self._waves:
            self._waves.add(wave)
            self._nodes.add(wave)

            if wave.parent is not None:
                self._factors.add(wave.parent)
                self._nodes.add(wave.parent)

    def register_measurement(self, measurement):
        """Register a measurement factor that has no output wave."""
        if measurement not in self._factors:
            self._factors.add(measurement)
            self._nodes.add(measurement)

    def compile(self):
        """
        Automatically register all Wave and Measurement objects
        defined as attributes of this Graph instance.
        """
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, Wave):
                self.register_wave(obj)
            elif isinstance(obj, Factor) and obj.output is None:
                self.register_measurement(obj)

        # Cache sorted node lists
        self._nodes_sorted = sorted(self._nodes, key=lambda x: x.generation)
        self._nodes_sorted_reverse = list(reversed(self._nodes_sorted))

    def forward(self):
        """Execute forward message passing in cached generation order."""
        for node in self._nodes_sorted:
            node.forward()

    def backward(self):
        """Execute backward message passing in reverse cached order."""
        for node in self._nodes_sorted_reverse:
            node.backward()

    def run(self, n_iter=10, callback=None, rng=None):
        """
        Run multiple rounds of belief propagation.

        Args:
            n_iter (int): Number of forward-backward iterations.
            callback (callable): Optional function called with (graph, t) at each step.
            rng (np.random.Generator or None): RNG used for initialization, defaults to internal.
        """
        rng = rng or self._rng
        for t in range(n_iter):
            self.forward()
            self.backward()
            if callback is not None:
                callback(self, t)

    def generate_sample(self, rng=None):
        """
        Generate a forward sample from the graphical model using all registered factors.
        Each factor's `generate_sample(rng)` is called in generation order.

        Args:
            rng (np.random.Generator or None): RNG used for sampling, defaults to internal.
        """
        rng = rng or self._rng
        factors = sorted(self._factors, key=lambda f: f.generation)
        for f in factors:
            f.generate_sample(rng)
            if hasattr(f, "update_observed_from_sample"):
                f.update_observed_from_sample()

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