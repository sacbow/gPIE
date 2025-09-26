import pytest
import numpy as np
import importlib.util

from gpie.core.backend import set_backend, np as gnp
from gpie.core.uncertain_array import UncertainArray
from gpie.graph.propagator.base import Propagator
from gpie.core.types import UnaryPropagatorPrecisionMode

# Optional CuPy
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp
    backend_libs = [np, cp]
else:
    backend_libs = [np]


class DummyPropagator(Propagator):
    """Minimal propagator for testing base class behavior."""

    def set_precision_mode_forward(self):
        return None

    def set_precision_mode_backward(self):
        return None

    def _compute_forward(self, inputs: dict[str, UncertainArray]) -> UncertainArray:
        # Just pass through the input
        return list(inputs.values())[0]

    def _compute_backward(self, output_msg: UncertainArray, exclude: str) -> UncertainArray:
        # Pass through the output
        return output_msg


@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_dtype_sync(xp):
    set_backend(xp)
    p = DummyPropagator(dtype=np.complex64)
    # dtype should switch to current backend dtype
    p.to_backend()
    assert p.dtype == gnp().dtype(np.complex64)


def test_set_precision_mode_accepts_enum_and_string():
    p = DummyPropagator()
    # string input
    p._set_precision_mode("scalar to array")
    assert isinstance(p.precision_mode_enum, UnaryPropagatorPrecisionMode)

    # reset new instance
    p2 = DummyPropagator()
    # enum input
    p2._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)
    assert isinstance(p2.precision_mode_enum, UnaryPropagatorPrecisionMode)


def test_set_precision_mode_rejects_invalid_string():
    p = DummyPropagator()
    with pytest.raises(ValueError):
        p._set_precision_mode("invalid_mode")


def test_set_precision_mode_rejects_invalid_type():
    p = DummyPropagator()
    with pytest.raises(TypeError):
        p._set_precision_mode(123)


def test_set_precision_mode_conflict():
    p = DummyPropagator()
    p._set_precision_mode("scalar")
    with pytest.raises(ValueError):
        p._set_precision_mode("array")
