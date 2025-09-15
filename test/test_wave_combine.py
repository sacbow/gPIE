import pytest
import numpy as np

from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.backend import set_backend, np as backend_np
from gpie.core.types import PrecisionMode
from gpie.graph.factor import Factor  # dummy base class

# Dummy factor class to simulate child connection
class DummyFactor(Factor):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"DummyFactor({self.name})"


@pytest.mark.parametrize("precision_mode", ["scalar", "array"])
def test_wave_combine_child_messages_correctness(precision_mode):
    set_backend(np)
    dtype = np.complex64
    shape = (32, 32)
    batch_size = 1
    num_children = 3

    # Create wave
    w = Wave(event_shape=shape, batch_size=batch_size, dtype=dtype, precision_mode=precision_mode)

    # Create dummy children and connect
    for i in range(num_children):
        f = DummyFactor(f"f{i}")
        w.add_child(f)

    # Create synthetic messages and inject
    for i, f in enumerate(w.children):
        mean = (i + 1) * backend_np().ones((batch_size, *shape), dtype=dtype)
        if precision_mode == "scalar":
            prec = 1.0
        else:
            prec = (i + 1) * backend_np().ones((batch_size, *shape), dtype=np.float32)
        ua = UncertainArray(mean, dtype=dtype, precision=prec)
        w.receive_message(f, ua)

    # Combine
    combined = w.combine_child_messages()
    assert combined is not None
    assert combined.data.shape == (batch_size, *shape)

    # Compute expected mean manually
    weighted_sum = backend_np().zeros((batch_size, *shape), dtype=dtype)
    total_prec = backend_np().zeros((batch_size, *shape), dtype=np.float32)

    for i in range(num_children):
        mu = (i + 1)
        if precision_mode == "scalar":
            p = 1.0
            p_array = backend_np().full((batch_size, *shape), p, dtype=np.float32)
        else:
            p = (i + 1)
            p_array = backend_np().full((batch_size, *shape), p, dtype=np.float32)

        weighted_sum += p_array * mu
        total_prec += p_array

    expected_mean = weighted_sum / total_prec
    abs_err = backend_np().abs(combined.data - expected_mean)
    rel_err = abs_err / (backend_np().abs(expected_mean) + 1e-8)

    assert backend_np().mean(rel_err) < 1e-4
    assert backend_np().max(rel_err) < 1e-3


def test_wave_precision_mode_mismatch_detection():
    set_backend(np)
    dtype = np.complex64
    shape = (16, 16)
    batch_size = 1

    w = Wave(event_shape=shape, batch_size=batch_size, dtype=dtype, precision_mode="scalar")

    # Connect two dummy children
    f1 = DummyFactor("f1")
    f2 = DummyFactor("f2")
    w.add_child(f1)
    w.add_child(f2)

    # First message: scalar precision
    mean1 = backend_np().ones((batch_size, *shape), dtype=dtype)
    ua1 = UncertainArray(mean1, dtype=dtype, precision=1.0)
    w.receive_message(f1, ua1)

    # Second message: array precision (mismatch)
    mean2 = backend_np().ones((batch_size, *shape), dtype=dtype) * 2.0
    prec2 = backend_np().full((batch_size, *shape), 3.0, dtype=np.float32)
    ua2 = UncertainArray(mean2, dtype=dtype, precision=prec2)
    w.receive_message(f2, ua2)

    # Combine should raise due to precision mode mismatch (assuming we enforce it)
    with pytest.raises(ValueError):
        _ = w.combine_child_messages()
