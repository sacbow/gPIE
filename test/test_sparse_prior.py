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


# ----------------------------------------------------------------------
# Backend-independence tests
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_approximate_posterior_backend_independence(xp):
    backend.set_backend(xp)

    batch_size = 4
    event_shape = (8, 8)
    shape = (batch_size,) + event_shape

    m = xp.random.normal(size=shape) + 1j * xp.random.normal(size=shape)
    precision = xp.ones(shape, dtype=xp.float32) * xp.float32(2.0)

    ua = UncertainArray(m, dtype=xp.complex64, precision=precision)
    prior = SparsePrior(rho=0.3, event_shape=event_shape, batch_size=batch_size, dtype=xp.complex64)

    # Explicitly set desired precision mode for posterior
    prior._set_precision_mode("array")

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
    assert xp.any(sample != 0.0)


# ----------------------------------------------------------------------
# Basic initialization & posterior properties
# ----------------------------------------------------------------------

def test_sparse_prior_initialization():
    backend.set_backend(np)

    sp = SparsePrior(rho=0.2, event_shape=(4,), batch_size=1, dtype=np.float32)
    assert sp.rho == np.float32(0.2)
    assert sp.output.event_shape == (4,)
    assert sp.output.batch_size == 1


def test_sparse_prior_approximate_posterior_real():
    backend.set_backend(np)

    sp = SparsePrior(rho=0.5, event_shape=(3,), dtype=np.float32)
    sp._set_precision_mode("scalar")

    incoming = UncertainArray(
        np.ones((1, 3), dtype=np.float32),
        precision=np.float32(2.0),
    )
    posterior = sp.approximate_posterior(incoming)
    assert posterior.data.shape == (1, 3)
    assert np.all(posterior.precision() > 0)


# ----------------------------------------------------------------------
# Damping behavior (full-batch)
# ----------------------------------------------------------------------

def test_sparse_prior_damping_effect_full_batch():
    """
    Verify that full-batch damping behaves as a convex combination of the
    undamped new message and the previous message.
    """
    backend.set_backend(np)

    event_shape = (3,)
    batch_size = 2

    incoming1 = UncertainArray(
        np.ones((batch_size, *event_shape), dtype=np.complex64),
        precision=np.float32(2.0),
    )
    incoming2 = UncertainArray(
        2.0 * np.ones((batch_size, *event_shape), dtype=np.complex64),
        precision=np.float32(2.0),
    )

    sp_ref = SparsePrior(
        rho=0.5,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.complex64,
        damping=0.0,
    )
    sp_ref._set_precision_mode("scalar")

    undamped1 = sp_ref._compute_message(incoming1, block=None)
    undamped2 = sp_ref._compute_message(incoming2, block=None)

    sp_damp = SparsePrior(
        rho=0.5,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.complex64,
        damping=0.5,
    )
    sp_damp._set_precision_mode("scalar")
    sp_damp._store_forward_message(undamped1)

    msg2 = sp_damp._compute_message(incoming2, block=None)
    expected = 0.5 * undamped2.data + 0.5 * undamped1.data

    assert np.allclose(msg2.data, expected, atol=1e-6)


# ----------------------------------------------------------------------
# Sampling & repr
# ----------------------------------------------------------------------

def test_sparse_prior_sampling_consistency():
    backend.set_backend(np)

    sp = SparsePrior(rho=0.3, event_shape=(5,), batch_size=2)
    rng = get_rng(seed=11)
    s1 = sp.get_sample_for_output(rng)
    rng = get_rng(seed=11)
    s2 = sp.get_sample_for_output(rng)

    assert np.array_equal(s1 == 0, s2 == 0)


def test_sparse_prior_repr():
    backend.set_backend(np)

    sp = SparsePrior(rho=0.1, event_shape=(2,), batch_size=1)
    r = repr(sp)
    assert "SPrior" in r
    assert "rho=0.1" in r


# ----------------------------------------------------------------------
# Block-wise behavior tests
# ----------------------------------------------------------------------

def test_sparse_prior_block_vs_full_batch_consistency():
    backend.set_backend(np)

    batch_size = 6
    event_shape = (4,)
    incoming = UncertainArray(
        np.random.randn(batch_size, *event_shape).astype(np.float32),
        precision=np.float32(2.0),
    ).as_array_precision()

    sp_full = SparsePrior(
        rho=0.5,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.float32,
        damping=0.0,
    )
    sp_full._set_precision_mode("array")
    msg_full = sp_full._compute_message(incoming, block=None)

    sp_block = SparsePrior(
        rho=0.5,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.float32,
        damping=0.0,
    )
    sp_block._set_precision_mode("array")

    blocks = [slice(0, 3), slice(3, 6)]
    assembled = np.empty_like(msg_full.data)

    for b in blocks:
        in_b = incoming.extract_block(b)
        out_b = sp_block._compute_message(in_b, block=b)
        assembled[b.start:b.stop] = out_b.data

    assert np.allclose(msg_full.data, assembled, atol=1e-5)


def test_sparse_prior_blockwise_damping_locality():
    backend.set_backend(np)

    batch_size = 4
    event_shape = (4,)

    incoming = UncertainArray(
        np.ones((batch_size, *event_shape), dtype=np.float32),
        precision=np.float32(2.0),
    )

    sp = SparsePrior(
        rho=0.5,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.float32,
        damping=0.5,
    )
    sp._set_precision_mode("scalar")
    sp.output_message = incoming

    prev_msg = UncertainArray(
        10.0 * np.ones((batch_size, *event_shape), dtype=np.float32),
        precision=np.float32(1.0),
    )
    sp._store_forward_message(prev_msg)

    block = slice(0, 2)
    
    prev_msg_data_copy = prev_msg.data.copy()
    sp.forward(block=block)
    updated = sp.last_forward_message.data
    
    assert not np.allclose(updated[0:2], prev_msg_data_copy[0:2])
    assert np.allclose(updated[2:4], prev_msg.data[2:4])


def test_sparse_prior_logZ_accumulation_blockwise():
    backend.set_backend(np)

    batch_size = 4
    event_shape = (2,)

    incoming = UncertainArray(
        np.random.randn(batch_size, *event_shape).astype(np.float32),
        precision=np.float32(2.0),
    )

    outgoing = UncertainArray(
        np.random.randn(batch_size, *event_shape).astype(np.float32),
        precision=np.float32(2.0),
    )

    sp = SparsePrior(
        rho=0.4,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.float32,
        damping="auto",
    )
    sp._set_precision_mode("scalar")

    sp.output_message = incoming
    sp._store_forward_message(outgoing)

    b1 = slice(0, 2)
    b2 = slice(2, 4)

    sp.forward(block=b1)
    assert sp.logZ != 0

    sp.forward(block=b2)
    assert sp.logZ == 0


def test_sparse_prior_dtype_integrity():
    backend.set_backend(np)

    batch_size = 3
    event_shape = (5,)

    incoming = UncertainArray(
        np.random.randn(batch_size, *event_shape).astype(np.float32),
        precision=np.float32(2.0),
    ).as_array_precision()

    sp = SparsePrior(
        rho=0.2,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.float32,
    )
    sp._set_precision_mode("array")

    msg = sp._compute_message(incoming, block=None)

    assert msg.data.dtype == np.float32
    assert msg.precision().dtype == np.float32


def test_sparse_prior_forward_sends_full_batch_shape():
    """
    Ensure block-wise forward sends a full-batch message every time.
    """
    backend.set_backend(np)

    batch_size = 5
    event_shape = (3,)

    incoming = UncertainArray(
        np.random.randn(batch_size, *event_shape).astype(np.float32),
        precision=np.float32(2.0),
    )

    outgoing = UncertainArray(
        np.random.randn(batch_size, *event_shape).astype(np.float32),
        precision=np.float32(2.0),
    )

    sp = SparsePrior(
        rho=0.3,
        event_shape=event_shape,
        batch_size=batch_size,
        dtype=np.float32,
    )
    sp._set_precision_mode("scalar")

    sp.output_message = incoming
    sp._store_forward_message(outgoing)

    sp.forward(block=slice(0, 2))

    assert sp.last_forward_message.data.shape == (batch_size, *event_shape)
