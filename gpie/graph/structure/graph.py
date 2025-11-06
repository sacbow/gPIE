from ..wave import Wave
from ..factor import Factor
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode
from ...core.rng_utils import get_rng
import contextlib
import threading
from typing import Literal, Optional

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
                
        # --- Step 6 : initialize messages ---
        for wave in self._waves:
            for factor in wave.children:
                wave.child_messages[factor] = UA.zeros(
                    event_shape=wave.event_shape,
                    batch_size=wave.batch_size,
                    dtype=wave.dtype,
                    precision=1.0,
                    scalar_precision=(wave.precision_mode_enum == PrecisionMode.SCALAR),
                )

    
    def to_backend(self) -> None:
        """
        Move all graph data (Waves, Factors, and their internal arrays) to the current backend (NumPy or CuPy),
        and resync RNGs. Sample arrays and observed values are delegated to node-local logic.
        """
        from ...core.rng_utils import get_rng

        # Waves
        for wave in self._waves:
            wave.to_backend()

        # Factors (Prior, Measurement, Propagator etc.)
        for factor in self._factors:
            factor.to_backend()

    
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
    
    def get_factor(self, label: str):
        """
        Retrieve the Factor instance with the given label.

        Args:
            label (str): Label assigned to the Factor.

        Returns:
            Factor instance with the specified label.

        Raises:
            ValueError: If no factor with the given label exists or if multiple factors share the label.
        """
        matches = [f for f in self._factors if getattr(f, "label", None) == label]
        if not matches:
            raise ValueError(f"No factor found with label '{label}'")
        if len(matches) > 1:
            raise ValueError(f"Multiple factors found with label '{label}'")
        return matches[0]
    

    def set_init_strategy(self, label: str, mode: str, data: Optional[np().ndarray] = None, verbose = True) -> None:
        """
        Set initialization strategy for the Prior associated with the given Wave label.

        Args:
            label (str): Label of the Wave node defined in the model DSL.
            mode (str): Initialization strategy ("uninformative", "sample", "manual").
            data (ndarray, optional): Manual initialization data (required if mode='manual').

        Raises:
            KeyError: If the given label does not correspond to any Wave.
            TypeError: If the corresponding parent node is not a Prior.
            ValueError: If mode is invalid or 'manual' is selected without data.
        """
        wave = self.get_wave(label)
        parent = wave.parent
        from ..prior.base import Prior 

        if not isinstance(parent, Prior):
            raise TypeError(f"Wave '{label}' is not generated by a Prior (found {type(parent).__name__}).")

        if mode == "manual":
            if data is None:
                raise ValueError("Manual initialization selected, but 'data' argument is missing.")
            parent.set_manual_init(data)
            parent.set_init_strategy("manual")
        else:
            parent.set_init_strategy(mode)

        if verbose:
            print(f"[Graph] Set init strategy for Prior '{type(parent).__name__}' (wave='{label}') → '{mode}'")


    def set_all_init_strategies(self, strategy_dict: dict[str, tuple[str, Optional[np().ndarray]]], verbose = True) -> None:
        """
        Set initialization strategies for multiple Priors at once.

        Args:
            strategy_dict (dict): A dictionary mapping label -> (mode, data)
                Example:
                    {
                        "x": ("manual", ndarray_x),
                        "y": ("uninformative", None),
                        "z": ("sample", None),
                    }

        Raises:
            ValueError: If an invalid mode is provided or data is missing for manual mode.
        """
        from ..prior.base import Prior
        count = 0

        for label, (mode, data) in strategy_dict.items():
            wave = self.get_wave(label)
            parent = wave.parent

            if parent is None or not isinstance(parent, Prior):
                raise TypeError(f"Wave '{label}' is not generated by a Prior node.")

            if mode == "manual":
                if data is None:
                    raise ValueError(f"Manual init for '{label}' requires ndarray data.")
                parent.set_manual_init(data)
                parent.set_init_strategy("manual")
            else:
                parent.set_init_strategy(mode)

            count += 1

        if verbose:
            print(f"[Graph] Applied initialization strategies for {count} Priors.")



    def forward(self):
        """Execute forward message passing in cached generation order."""
        for node in self._nodes_sorted:
            node.forward()

    def backward(self):
        """Execute backward message passing in reverse cached order."""
        for node in self._nodes_sorted_reverse:
            node.backward()

    def run(self, n_iter=10, callback=None, verbose=False):
        """
        Run multiple rounds of belief propagation with optional progress bar.

        Args:
            n_iter (int): Number of forward-backward iterations.
            callback (callable or None): Function called as callback(graph, t).
            rng (np.random.Generator or None): Optional RNG.
            verbose (bool): Whether to show progress bar (requires tqdm).
        """

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


    def generate_sample(self, rng=None, update_observed: bool = True, mask: Optional[np().ndarray] = None):
        """
        Generate a full sample from the generative model defined by the graph.

        Args:
            rng: RNG used for latent and observed sampling (optional).
            update_observed: If True, observed data is updated from the sample.
            mask: Optional mask to apply to Measurement nodes during observation update.
        """
        rng = rng or get_rng()

        # 1. Generate latent samples
        for node in self._nodes_sorted:
            if isinstance(node, Wave):
                node._generate_sample(rng=rng)

        # 2. Generate noisy observed samples
        for meas in self._factors:
            if hasattr(meas, "_generate_sample") and callable(meas._generate_sample):
                meas._generate_sample(rng)

        # 3. Promote to observed (with optional mask)
        if update_observed:
            for meas in self._factors:
                if hasattr(meas, "update_observed_from_sample"):
                    meas.update_observed_from_sample(mask=mask)



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
    

    def to_networkx(self) -> "nx.DiGraph":
        import networkx as nx

        G = nx.DiGraph()
        for wave in self._waves:
            G.add_node(
                id(wave),
                label=(wave.label if getattr(wave, "label", None) else "Wave"),
                type="wave",
                ref=wave,
            )
        for factor in self._factors:
            factor_label = getattr(factor, "label", None)
            if not isinstance(factor_label, str) or not factor_label:
                factor_label = factor.__class__.__name__

            G.add_node(
                id(factor),
                label=factor_label,
                type="factor",
                ref=factor,
            )
            for wave in factor.inputs.values():
                G.add_edge(id(wave), id(factor))
            if getattr(factor, "output", None):
                G.add_edge(id(factor), id(factor.output))
        return G


    def visualize(
        self,
        backend: str = "bokeh",
        layout: str = "graphviz",
        output_path: Optional[str] = None
        ):
        from .visualization import visualize_graph
        return visualize_graph(self, backend=backend, layout=layout, output_path=output_path)
