import pytest
import numpy as np

from gpie.graph.prior.base import Prior
from gpie.core import backend
from gpie.core.types import PrecisionMode
from gpie.core.uncertain_array import UncertainArray

backend.set_backend(np)

class DummyPrior(Prior):
    def _compute_message(self, incoming: UncertainArray) -> UncertainArray:
        # Return incoming unchanged for test purpose
        return incoming

def test_prior_initialization_and_wave_access():
    p = DummyPrior(shape=(4, 4), precision_mode=PrecisionMode.SCALAR)
    assert p.shape == (4, 4)
    assert isinstance(~p, type(p.output))  # __invert__ works
    assert p.output.shape == (4, 4)
    assert p.precision_mode == PrecisionMode.SCALAR  


def test_prior_forward_message_generation():
    p = DummyPrior(shape=(4, 4), precision_mode=PrecisionMode.ARRAY)
    rng = np.random.default_rng(seed=123)
    p.set_init_rng(rng)

    assert p.output_message is None
    p.forward()
    msg = p.output_message = p.output.parent_message
    assert isinstance(msg, UncertainArray)
    assert msg.shape == (4, 4)
    assert msg.dtype == np.complex128

def test_prior_forward_with_existing_message_calls_compute():
    class CountingPrior(DummyPrior):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.compute_called = False

        def _compute_message(self, incoming):
            self.compute_called = True
            return incoming

    p = CountingPrior(shape=(2, 2), precision_mode=PrecisionMode.SCALAR)
    p.set_init_rng(np.random.default_rng())

    # First call: initialize output message
    p.forward()
    assert p.output_message is None  # still None, but parent_message is updated

    # Manually fake a message
    p.output_message = UncertainArray(np.ones((2, 2)), precision=1.0)

    # Second call: should invoke _compute_message
    p.forward()
    assert p.compute_called
