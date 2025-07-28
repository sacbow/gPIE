import importlib.util
import pytest

import numpy as np
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

from gpie.core import backend
from gpie.graph.prior.sparse_prior import SparsePrior
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_approximate_posterior_backend_independence(xp):
    backend.set_backend(xp)

    shape = (8, 8)
    m = xp.random.normal(size=shape) + 1j * xp.random.normal(size=shape)
    precision = xp.ones(shape) * 2.0

    ua = UncertainArray(m, dtype=xp.complex128, precision=precision)
    prior = SparsePrior(rho=0.3, shape=shape, dtype=xp.complex128)
    posterior = prior.approximate_posterior(ua)

    # Check output type and shape
    assert isinstance(posterior, UncertainArray)
    assert posterior.data.shape == shape
    assert posterior.precision(raw=True).shape == shape or posterior.precision(raw=True).ndim == 0

    # Ensure output does not contain NaNs or infs
    assert not xp.any(xp.isnan(posterior.data))
    assert not xp.any(xp.isinf(posterior.data))


@pytest.mark.parametrize("xp", backend_libs)
def test_generate_sample_backend_independence(xp):
    backend.set_backend(xp)
    prior = SparsePrior(rho=0.5, shape=(8, 8), dtype=xp.complex128)
    prior.generate_sample(rng=xp.random.default_rng())
    sample = prior.output.get_sample()

    assert sample.shape == (8, 8)
    assert sample.dtype == xp.complex128
    assert xp.any(sample != 0.0)  # nonzero elements should exist

def test_sparse_prior_initialization():
    sp = SparsePrior(rho=0.2, shape=(4,), dtype=np.float64)
    assert sp.rho == 0.2
    assert sp.output.shape == (4,)

def test_sparse_prior_approximate_posterior_real():
    sp = SparsePrior(rho=0.5, shape=(3,), dtype=np.float64)
    incoming = UncertainArray(np.ones((3,), dtype=np.float64), precision=2.0)
    posterior = sp.approximate_posterior(incoming)
    assert posterior.data.shape == (3,)
    assert np.all(posterior.precision() > 0)

def test_sparse_prior_damping_effect():
    sp = SparsePrior(rho=0.5, shape=(3,), precision_mode=PrecisionMode.SCALAR)
    incoming = UncertainArray(np.ones((3,), dtype=np.complex128), precision=2.0)
    msg1 = sp._compute_message(incoming)
    sp.damping = 0.5
    msg2 = sp._compute_message(incoming)
    assert np.allclose(msg2.data, 0.5 * msg1.data + 0.5 * sp.old_msg.data)

def test_sparse_prior_sampling_consistency():
    sp = SparsePrior(rho=0.3, shape=(5,))
    rng = np.random.default_rng(seed=11)
    s1 = sp.get_sample_for_output(rng)
    rng = np.random.default_rng(seed=11)
    s2 = sp.get_sample_for_output(rng)
    assert np.array_equal(s1 == 0, s2 == 0)


def test_sparse_prior_repr():
    sp = SparsePrior(rho=0.1, shape=(2,))
    r = repr(sp)
    assert "SPrior" in r
    assert "rho=0.1" in r
