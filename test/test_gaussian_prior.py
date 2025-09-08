import pytest
import numpy as np
from gpie.graph.prior.gaussian_prior import GaussianPrior
from gpie.core.types import PrecisionMode
from gpie.core.uncertain_array import UncertainArray


def test_gaussian_prior_initialization():
    gp = GaussianPrior(var=2.0, event_shape=(3, 3), batch_size=5, dtype=np.complex128)
    assert gp.var == 2.0
    assert gp.precision == 0.5
    assert gp.output.event_shape == (3, 3)
    assert gp.output.batch_size == 5
    assert gp.output.dtype == np.complex128


def test_gaussian_prior_compute_message_scalar():
    gp = GaussianPrior(
        var=1.0,
        event_shape=(2, 2),
        batch_size=3,
        precision_mode=PrecisionMode.SCALAR,
    )
    msg = gp._compute_message(UncertainArray.zeros(event_shape=(2, 2), batch_size=3))
    assert isinstance(msg, UncertainArray)
    assert msg.precision_mode == PrecisionMode.SCALAR
    assert msg.precision().shape == (3, 2, 2)
    assert np.allclose(msg.precision(), 1.0)


def test_gaussian_prior_compute_message_array():
    gp = GaussianPrior(
        var=2.0,
        event_shape=(2, 2),
        batch_size=4,
        precision_mode=PrecisionMode.ARRAY,
    )
    assert gp.output._precision_mode == PrecisionMode.ARRAY
    msg = gp._compute_message(UncertainArray.zeros(event_shape=(2, 2), batch_size=4))
    assert msg.precision_mode == PrecisionMode.ARRAY
    assert msg.precision().shape == (4, 2, 2)
    assert np.allclose(msg.precision(), 1 / 2.0)


def test_gaussian_prior_get_sample_for_output():
    gp = GaussianPrior(var=4.0, event_shape=(2, 2), batch_size=6)
    s = gp.get_sample_for_output()
    assert s.shape == (6, 2, 2)
    assert s.dtype == gp.dtype
    assert np.allclose(s.mean(), 0, atol=1.0)  # mean ≈ 0
    assert np.allclose(np.var(s), gp.var, rtol=0.5)  # var ≈ gp.var (allowing some noise)


def test_gaussian_prior_repr():
    gp = GaussianPrior(var=1.0, event_shape=(1,))
    r = repr(gp)
    assert "GaussianPrior" in r
    assert "var=1.0" in r
