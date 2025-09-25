import importlib.util
import numpy as np
import pytest

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
if cupy_spec is not None:
    import cupy as cp
    backend_libs = [np, cp]
else:
    cp = None
    backend_libs = [np]


class DummyFactor:
    def __init__(self):
        self.received = None

    def receive_message(self, wave, msg):
        self.received = (wave, msg)

    def get_input_precision_mode(self, wave):
        return "scalar"

    def get_output_precision_mode(self):
        return "scalar"


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_sample_and_clear(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(4, 4), dtype=xp.complex64)
    sample = xp.ones((1, 4, 4), dtype=xp.complex64)
    w.set_sample(sample)
    assert xp.allclose(w.get_sample(), sample)
    w.clear_sample()
    assert w.get_sample() is None


@pytest.mark.parametrize("xp", backend_libs)
def test_receive_message_and_compute_belief_scalar(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(2, 2), batch_size=1, dtype=xp.complex64)
    w._set_precision_mode("scalar")
    parent = DummyFactor()
    child1 = DummyFactor()
    child2 = DummyFactor()

    w.set_parent(parent)
    w.add_child(child1)
    w.add_child(child2)

    msg1 = UncertainArray(xp.full((1, 2, 2), 1.0), precision=1.0)
    msg2 = UncertainArray(xp.full((1, 2, 2), 3.0), precision=1.0)
    parent_msg = UncertainArray(xp.full((1, 2, 2), 2.0), precision=2.0)

    w.receive_message(child1, msg1)
    w.receive_message(child2, msg2)
    w.receive_message(parent, parent_msg)

    belief = w.compute_belief()
    assert isinstance(belief, UncertainArray)
    assert belief.event_shape == (2, 2)
    assert xp.allclose(belief.data.shape, (1, 2, 2))


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_to_backend_converts_all_messages(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(2, 2), batch_size=2, dtype=xp.complex64)
    w._set_precision_mode("array")
    child = DummyFactor()
    parent = DummyFactor()
    w.add_child(child)
    w.set_parent(parent)

    msg = UncertainArray(xp.full((2, 2, 2), 1.0), precision=xp.ones((2, 2, 2)))
    parent_msg = UncertainArray(xp.full((2, 2, 2), 1.0), precision=xp.ones((2, 2, 2)))
    w.receive_message(child, msg)
    w.receive_message(parent, parent_msg)
    belief = w.compute_belief()

    if cp is not None and xp.__name__ == "numpy":
        backend.set_backend(cp)
        w.to_backend()
        assert isinstance(w.belief.data, cp.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, cp.ndarray)
    elif cp is None and xp.__name__ == "numpy":
        pytest.skip("CuPy not available, skipping transfer-to-backend test")
    else:
        backend.set_backend(np)
        w.to_backend()
        assert isinstance(w.belief.data, np.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, np.ndarray)


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_to_backend_converts_all_messages_with_parent(xp):
    backend.set_backend(xp)
    w = Wave(event_shape=(2, 2), batch_size=2, dtype=xp.complex64)
    w._set_precision_mode("array")
    parent = DummyFactor()
    child = DummyFactor()
    w.set_parent(parent)
    w.add_child(child)

    msg = UncertainArray(xp.full((2, 2, 2), 1.0), precision=xp.ones((2, 2, 2)))
    w.receive_message(child, msg)
    w.receive_message(parent, msg)

    w.compute_belief()

    if cp is not None and xp.__name__ == "numpy":
        backend.set_backend(cp)
        w.to_backend()
        assert isinstance(w.belief.data, cp.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, cp.ndarray)
    elif cp is None and xp.__name__ == "numpy":
        pytest.skip("CuPy not available, skipping transfer-to-backend test")
    else:
        backend.set_backend(np)
        w.to_backend()
        assert isinstance(w.belief.data, np.ndarray)
        for m in w.child_messages.values():
            assert isinstance(m.data, np.ndarray)
