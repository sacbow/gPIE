import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.core.accumulative_uncertain_array import AccumulativeUncertainArray
from gpie.core.uncertain_array import UncertainArray

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_repr_and_clear(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [(slice(0, 2), slice(0, 2))]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with one patch: fill with ones, precision=1
    ua = UncertainArray.zeros((2, 2), batch_size=1, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    assert xp.any(aua.precision != 0)
    assert xp.any(aua.weighted_data != 0)

    # clear must reset both arrays
    aua.clear()
    assert xp.allclose(aua.precision, 0)
    assert xp.allclose(aua.weighted_data, 0)

    # repr sanity check
    rep = repr(aua)
    assert "AUA" in rep and "event_shape" in rep


@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_mul_and_extract_patches(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with two patches: all zeros, precision=1
    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    aua.scatter_mul(ua)

    # extract_patches should return same UA back
    ua_out = aua.extract_patches()
    assert ua_out.batch_size == 2
    assert ua_out.event_shape == (2, 2)
    assert xp.allclose(ua_out.data, xp.zeros_like(ua_out.data))
    assert xp.allclose(ua_out.precision(raw=False), xp.ones_like(ua_out.data))


@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_mul_with_overlapping_indices(xp):
    backend.set_backend(xp)
    event_shape = (3, 3)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(1, 3), slice(1, 3)),  # overlaps with first
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with two patches: all ones, precision=1
    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    # Overlap at (1,1) should accumulate precision=2, weighted=2
    total_prec = aua.precision[1, 1]
    total_weighted = aua.weighted_data[1, 1]
    assert xp.allclose(total_prec, 2.0)
    assert xp.allclose(total_weighted, 2.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_as_uncertain_array(xp):
    backend.set_backend(xp)
    event_shape = (2, 2)
    indices = [(slice(0, 2), slice(0, 2))]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with one patch: ones, precision=2
    ua = UncertainArray.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=2.0)
    ua.data[...] = 1.0
    aua.scatter_mul(ua)

    ua_out = aua.as_uncertain_array()
    assert ua_out.batch_size == 1
    assert ua_out.event_shape == (2, 2)
    assert xp.allclose(ua_out.data, xp.ones_like(ua_out.data))
    assert xp.allclose(ua_out.precision(raw=False), xp.full((1, 2, 2), 2.0))


@pytest.mark.parametrize("xp", backend_libs)
def test_mul_ua(xp):
    backend.set_backend(xp)
    event_shape = (3, 3)
    indices = [(slice(0, 3), slice(0, 3))]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with batch_size=1, event_shape=(3,3), precision=2, data=1
    ua = UncertainArray.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=2.0)
    ua.data[...] = 1.0

    aua.mul_ua(ua)

    # precision must be filled with 2.0
    assert xp.allclose(aua.precision, xp.full(event_shape, 2.0))
    # weighted_data must be data * precision = 1*2 = 2
    assert xp.allclose(aua.weighted_data, xp.full(event_shape, 2.0))

    # apply mul_ua again → should accumulate
    aua.mul_ua(ua)
    assert xp.allclose(aua.precision, xp.full(event_shape, 4.0))
    assert xp.allclose(aua.weighted_data, xp.full(event_shape, 4.0))



