import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.prior.support_prior import SupportPrior
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_initialization_and_backend(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])
    prior = SupportPrior(support=support, dtype=xp.complex128)

    assert prior.support.shape == (2, 2)
    assert prior.support.dtype == bool
    assert prior.output.shape == (2, 2)

    # Test to_backend conversion (simulate backend switch)
    new_backend = cp if xp is np else np
    backend.set_backend(new_backend)
    prior.to_backend()
    assert isinstance(prior.support, new_backend.ndarray)
    assert prior.support.dtype == bool
    assert prior._fixed_msg_array.data.shape == (2, 2)


@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_compute_message_array_mode(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])
    prior = SupportPrior(support=support, dtype=xp.complex128, precision_mode=PrecisionMode.ARRAY)

    incoming = UncertainArray.zeros((2, 2), dtype=xp.complex128, precision=1.0)
    msg = prior._compute_message(incoming)

    assert isinstance(msg, UncertainArray)
    assert msg.precision_mode == PrecisionMode.ARRAY
    assert xp.allclose(msg.data, xp.zeros((2, 2), dtype=xp.complex128))
    assert msg.precision().shape == (2, 2)


@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_compute_message_scalar_mode(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])
    prior = SupportPrior(support=support, dtype=xp.complex128, precision_mode=PrecisionMode.SCALAR)

    incoming = UncertainArray.zeros((2, 2), dtype=xp.complex128, precision=1.0)
    msg = prior._compute_message(incoming)

    assert isinstance(msg, UncertainArray)
    assert msg.precision_mode == PrecisionMode.SCALAR
    assert msg.data.shape == (2, 2)

@pytest.mark.parametrize("xp", backend_libs)
def test_support_prior_sampling(xp):
    backend.set_backend(xp)
    support = xp.array([[True, False], [False, True]])
    prior = SupportPrior(support=support, dtype=xp.complex128)

    prior.generate_sample(rng=xp.random.default_rng())
    sample = prior.output.get_sample()
    assert sample.shape == (2, 2)
    assert xp.all(sample[~support] == 0)

    rng = xp.random.default_rng(seed=123)
    s1 = prior.get_sample_for_output(rng)
    rng = xp.random.default_rng(seed=123)
    s2 = prior.get_sample_for_output(rng)

    if xp.__name__ == "numpy":
        assert xp.allclose(s1, s2)  
    else:
        assert s1.shape == s2.shape 



def test_support_prior_repr():
    support = np.array([True, False, True])
    prior = SupportPrior(support=support, dtype=np.complex128)
    r = repr(prior)
    assert "SupportPrior" in r
    assert "mode=" in r
