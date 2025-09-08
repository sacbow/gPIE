import importlib.util
import pytest
import numpy as np

cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

from gpie.core import backend
from gpie.core.rng_utils import get_rng
from gpie.graph.prior.sparse_prior import SparsePrior
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_approximate_posterior_backend_independence(xp):
    backend.set_backend(xp)

    batch_size = 4
    event_shape = (8, 8)
    shape = (batch_size,) + event_shape

    m = xp.random.normal(size=shape) + 1j * xp.random.normal(size=shape)
    precision = xp.ones(shape) * 2.0

    ua = UncertainArray(m, dtype=xp.complex64, precision=precision)
    prior = SparsePrior(rho=0.3, event_shape=event_shape, batch_size=batch_size, dtype=xp.complex64)
    posterior = prior.approximate_posterior(ua)

    assert isinstance(posterior, UncertainArray)
    assert posterior.data.shape == shape
    assert posterior.precision(raw=True).shape == shape or posterior._scalar_precision

    assert not xp.any(xp.isnan(posterior.data))
    assert not xp.any(xp.isinf(posterior.data))


@pytest.mark.parametrize("xp", backend_libs)
def test_sparse_prior_sample_backend_independence(xp):
    backend.set_backend(xp)

    prior = SparsePrior(rho=0.5, event_shape=(8, 8), batch_size=3, dtype=xp.complex64)
    sample = prior.get_sample_for_output(rng=get_rng())

    assert sample.shape == (3, 8, 8)
    assert sample.dtype == xp.complex64
    assert xp.any(sample != 0.0)  # ensure some nonzero elements


def test_sparse_prior_initialization():
    sp = SparsePrior(rho=0.2, event_shape=(4,), batch_size=1, dtype=np.float32)
    assert sp.rho == np.float32(0.2)
    assert sp.output.event_shape == (4,)
    assert sp.output.batch_size == 1


def test_sparse_prior_approximate_posterior_real():
    sp = SparsePrior(rho=0.5, event_shape=(3,), dtype=np.float32)
    incoming = UncertainArray(np.ones((1, 3), dtype=np.float32), precision=2.0)
    posterior = sp.approximate_posterior(incoming)
    assert posterior.data.shape == (1, 3)
    assert np.all(posterior.precision() > 0)


def test_sparse_prior_damping_effect():
    sp = SparsePrior(rho=0.5, event_shape=(3,), precision_mode=PrecisionMode.SCALAR)
    incoming = UncertainArray(np.ones((1, 3), dtype=np.complex64), precision=2.0)
    msg1 = sp._compute_message(incoming)
    sp.damping = 0.5
    msg2 = sp._compute_message(incoming)
    assert np.allclose(msg2.data, 0.5 * msg1.data + 0.5 * sp.old_msg.data)


def test_sparse_prior_sampling_consistency():
    sp = SparsePrior(rho=0.3, event_shape=(5,), batch_size=2)
    rng = get_rng(seed=11)
    s1 = sp.get_sample_for_output(rng)
    rng = get_rng(seed=11)
    s2 = sp.get_sample_for_output(rng)
    assert np.array_equal(s1 == 0, s2 == 0)


def test_sparse_prior_repr():
    sp = SparsePrior(rho=0.1, event_shape=(2,), batch_size=1)
    r = repr(sp)
    assert "SPrior" in r
    assert "rho=0.1" in r
