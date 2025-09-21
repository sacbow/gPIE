from functools import wraps
from .graph import Graph

def model(func):
    """
    Decorator that wraps a model definition function and returns a compiled Graph.

    Usage:
        @model
        def my_model():
            x = ~SparsePrior(rho = 0.1, shape = (128,128), label = "obj)
            with observe():
                GaussianMeasurement(var = 0.1) @ x
            return 

        g = my_model()
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        g = Graph()
        with g.observe():
            func(*args, **kwargs)
        g.compile()
        return g
    return wrapper

def observe():
    g = Graph.get_active_graph()
    if g is None:
        raise RuntimeError("observe() must be called inside a Graph context.")
    return g.observe()
