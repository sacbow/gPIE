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
    cp = None  # define cp = None so it won't cause NameError
    backend_libs = [np]


@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_initialization_and_backend(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])  # shape = (2, 2)
    event_shape = (2, 2)
    batch_size = 2

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=xp.complex64
    )

    assert prior.support.shape == (2, 2, 2)
    assert prior.support.dtype == bool
    assert prior.output.event_shape == (2, 2)
    assert prior.output.batch_size == 2

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
    assert prior._fixed_msg_array.data.shape == (2, 2, 2)


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
        precision_mode=PrecisionMode.ARRAY
    )

    ua = UncertainArray.zeros(
        event_shape=event_shape,
        batch_size=3,
        dtype=xp.complex64,
        precision=1.0
    )
    msg = prior._compute_message(ua)

    assert isinstance(msg, UncertainArray)
    assert msg.precision_mode == PrecisionMode.ARRAY
    assert msg.data.shape == (3, 2, 2)
    assert msg.precision().shape == (3, 2, 2)


@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_compute_message_scalar_mode(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])
    event_shape = (2, 2)

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        batch_size=1,
        dtype=xp.complex64,
        precision_mode=PrecisionMode.SCALAR
    )

    ua = UncertainArray.zeros(
        event_shape=event_shape,
        batch_size=1,
        dtype=xp.complex64,
        precision=1.0
    )
    msg = prior._compute_message(ua)

    assert isinstance(msg, UncertainArray)
    assert msg.precision_mode == PrecisionMode.SCALAR
    assert msg.data.shape == (1, 2, 2)


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
        dtype=xp.complex64
    )

    rng = get_rng(seed=42)
    s1 = prior.get_sample_for_output(rng)

    rng = get_rng(seed=42)
    s2 = prior.get_sample_for_output(rng)

    assert s1.shape == (2, 2, 2)
    assert s2.shape == (2, 2, 2)
    assert xp.all(s1[~prior.support] == 0)
    assert xp.all(s2[~prior.support] == 0)

    if xp.__name__ == "numpy":
        assert xp.allclose(s1, s2)


def test_support_prior_repr():
    backend.set_backend(np)
    support = np.array([[True, False], [False, True]])
    event_shape = (2, 2)

    prior = SupportPrior(
        support=support,
        event_shape=event_shape,
        dtype=np.complex64
    )

    r = repr(prior)
    assert "SupportPrior" in r
    assert "mode=" in r
