from graph.wave import Wave
from graph.factor import Factor


class Graph:
    def __init__(self):
        self._nodes = set()
        self._waves = set()
        self._factors = set()

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

    def forward(self):
        """Execute forward message passing in generation order."""
        for node in sorted(self._nodes, key=lambda x: x.generation):
            node.forward()

    def backward(self):
        """Execute backward message passing in reverse generation order."""
        for node in sorted(self._nodes, key=lambda x: -x.generation):
            node.backward()

    def run(self, n_iter=10, callback=None):
        """
        Run multiple rounds of belief propagation.

        Args:
            n_iter (int): Number of forward-backward iterations.
            callback (callable): Optional function called with (graph, t) at each step.
        """
        for t in range(n_iter):
            self.forward()
            self.backward()
            if callback is not None:
                callback(self, t)

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

        # Select layout
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

        # Separate node types
        node_shapes = {'circle': [], 'square': []}
        for node in G.nodes:
            shape = G.nodes[node]['shape']
            node_shapes[shape].append(node)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                           nodelist=node_shapes['circle'],
                           node_shape='o', node_color='skyblue', label='Wave')
        nx.draw_networkx_nodes(G, pos,
                           nodelist=node_shapes['square'],
                           node_shape='s', node_color='lightgreen', label='Factor')

        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)

        # Draw labels
        if with_labels:
            labels = {n: G.nodes[n]['label'] for n in G.nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=font_size)

        plt.title("Computational Factor Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()