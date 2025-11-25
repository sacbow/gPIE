import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.graph.prior.support_prior import SupportPrior
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


# ----------------------------------------------------------------------
# Initialization + Backend conversion
# ----------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_initialization_and_backend(xp):
    backend.set_backend(xp)

    support = xp.array([[True, False], [False, True]])
    event_shape = (2, 2)
    batch_size = 2

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=xp.complex64,
    )

    # Support is broadcast to (B,H,W)
    assert prior.support.shape == (2, 2, 2)
    assert prior.support.dtype == bool
    assert prior.output.event_shape == (2, 2)
    assert prior.output.batch_size == 2

    # msg shape
    assert prior.const_msg.data.shape == (2, 2, 2)
    assert prior.const_msg.precision().shape == (2, 2, 2)

    # to_backend test
    if cp is not None and xp.__name__ == "numpy":
        backend.set_backend(cp)
        prior.to_backend()
        assert isinstance(prior.support, cp.ndarray)
    elif cp is None and xp.__name__ == "numpy":
        pytest.skip("CuPy not available, skipping transfer-to-backend test")
    else:
        backend.set_backend(np)
        prior.to_backend()
        assert isinstance(prior.support, np.ndarray)

    assert prior.support.dtype == bool
    assert prior.const_msg.data.shape == (2, 2, 2)


# ----------------------------------------------------------------------
# compute_message (always returns const_msg)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_compute_message_array_mode(xp):
    backend.set_backend(xp)

    support = xp.array([[True, False], [False, True]])
    event_shape = (2, 2)

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        batch_size=3,
        dtype=xp.complex64,
    )

    ua = UncertainArray.zeros(
        event_shape=event_shape,
        batch_size=3,
        dtype=xp.complex64,
        precision=1.0,
    )

    msg = prior._compute_message(ua)
    assert isinstance(msg, UncertainArray)
    assert msg.data.shape == (3, 2, 2)
    assert msg.precision().shape == (3, 2, 2)

    # equals cached const_msg (up to batch dimension)
    assert xp.allclose(msg.data, prior.const_msg.data)
    assert xp.allclose(msg.precision(), prior.const_msg.precision())


# ----------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_sampling_and_determinism(xp):
    backend.set_backend(xp)

    support = xp.array([[True, False], [False, True]])
    event_shape = (2, 2)
    batch_size = 2

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=xp.complex64,
    )

    rng = get_rng(seed=42)
    s1 = prior.get_sample_for_output(rng)

    rng = get_rng(seed=42)
    s2 = prior.get_sample_for_output(rng)

    assert s1.shape == (2, 2, 2)
    assert s2.shape == (2, 2, 2)

    # outside support must be zero
    assert xp.all(s1[~prior.support] == 0)
    assert xp.all(s2[~prior.support] == 0)

    # deterministic under fixed seed
    if xp.__name__ == "numpy":
        assert xp.allclose(s1, s2)


# ----------------------------------------------------------------------
# block-wise forward must be identical to full forward
# ----------------------------------------------------------------------
def test_support_prior_forward_block_agnostic():
    backend.set_backend(np)

    support = np.array([[True, False], [False, True]])
    event_shape = (2, 2)
    batch_size = 3

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.complex64,
    )

    # simulate first iteration
    rng = get_rng(seed=0)
    prior.set_init_rng(rng)
    prior.forward(block=None)
    init_msg = prior.last_forward_message

    # simulate later iteration with blocks
    # setup a fake incoming message
    ua = UncertainArray.zeros(event_shape = event_shape, batch_size = batch_size, dtype=np.complex64)
    prior.output_message = ua

    # block-wise
    prior.forward(block=slice(0, 1))
    msg1 = prior.last_forward_message.data.copy()

    prior.forward(block=slice(1, 3))
    msg2 = prior.last_forward_message.data.copy()

    # both should equal const_msg everywhere
    assert np.allclose(msg1, prior.const_msg.data)
    assert np.allclose(msg2, prior.const_msg.data)


# ----------------------------------------------------------------------
# repr
# ----------------------------------------------------------------------
def test_support_prior_repr():
    backend.set_backend(np)

    support = np.array([[True, False], [False, True]])
    event_shape = (2, 2)

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        dtype=np.complex64,
    )

    r = repr(prior)
    assert "SupportPrior" in r
    assert "mode=ARRAY" in r
