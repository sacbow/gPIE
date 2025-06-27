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
