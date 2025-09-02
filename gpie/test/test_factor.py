import pytest
from gpie.graph.wave import Wave
from gpie.graph.factor import Factor
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

# Minimal concrete subclass for testing
class DummyFactor(Factor):
    def forward(self):
        pass

    def backward(self):
        pass


def test_initialization():
    f = DummyFactor()
    assert f.inputs == {}
    assert f.output is None
    assert f.input_messages == {}
    assert f.output_message is None
    assert f.precision_mode is None
    assert f.generation is None


def test_add_input_and_connect_output_sets_generations():
    w1 = Wave((2, 2))
    w2 = Wave((2, 2))
    f = DummyFactor()

    w1._set_generation(3)
    f.add_input("x", w1)
    f.connect_output(w2)

    assert f.generation == 4
    assert w2.generation == 5
    assert w2.parent == f
    assert w1 in f.input_messages


def test_set_precision_mode():
    f = DummyFactor()
    f._set_precision_mode("scalar")
    assert f.precision_mode == PrecisionMode.SCALAR

    # Should raise ValueError on conflicting mode
    with pytest.raises(ValueError):
        f._set_precision_mode("array")

    # Should raise on invalid type
    with pytest.raises(TypeError):
        f._set_precision_mode(123)


def test_receive_message_routing():
    f = DummyFactor()
    w_in = Wave((2, 2))
    w_out = Wave((2, 2))
    msg = UncertainArray(array=[[1.0, 1.0], [1.0, 1.0]], precision=1.0)

    f.add_input("x", w_in)
    f.connect_output(w_out)

    f.receive_message(w_in, msg)
    assert f.input_messages[w_in] is msg

    f.receive_message(w_out, msg)
    assert f.output_message is msg

    # Invalid wave
    w_unlinked = Wave((2, 2))
    with pytest.raises(ValueError):
        f.receive_message(w_unlinked, msg)
