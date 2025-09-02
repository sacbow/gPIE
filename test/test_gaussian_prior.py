import pytest
import numpy as np
from gpie.graph.prior.gaussian_prior import GaussianPrior
from gpie.core.types import PrecisionMode
from gpie.core.uncertain_array import UncertainArray

def test_gaussian_prior_initialization():
    gp = GaussianPrior(var=2.0, shape=(3, 3), dtype=np.complex128)
    assert gp.var == 2.0
    assert gp.precision == 0.5
    assert gp.output.shape == (3, 3)
    assert gp.output.dtype == np.complex128

def test_gaussian_prior_compute_message_scalar():
    gp = GaussianPrior(var=1.0, shape=(2, 2), precision_mode=PrecisionMode.SCALAR)
    msg = gp._compute_message(UncertainArray.zeros((2, 2)))
    assert isinstance(msg, UncertainArray)
    assert msg.precision_mode == PrecisionMode.SCALAR
    assert np.allclose(msg.precision(), 1.0)

def test_gaussian_prior_compute_message_array():
    gp = GaussianPrior(var=2.0, shape=(2, 2), precision_mode=PrecisionMode.ARRAY)
    assert gp.output._precision_mode == PrecisionMode.ARRAY
    msg = gp._compute_message(UncertainArray.zeros((2, 2)))
    assert msg.precision_mode == PrecisionMode.ARRAY
    assert np.allclose(msg.precision(), 1/2.0)

def test_gaussian_prior_get_sample_for_output_and_generate_sample():
    gp = GaussianPrior(var=4.0, shape=(2, 2))
    s1 = gp.get_sample_for_output()
    assert s1.shape == (2, 2)
    assert s1.dtype == gp.dtype
    # generate_sample should call set_sample with similar data
    gp.generate_sample(rng=np.random.default_rng(0))
    assert gp.output.get_sample() is not None
    assert gp.output.get_sample().shape == (2, 2)

def test_gaussian_prior_repr():
    gp = GaussianPrior(var=1.0, shape=(1,))
    r = repr(gp)
    assert "GaussianPrior" in r
    assert "var=1.0" in r
