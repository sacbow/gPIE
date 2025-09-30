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
def test_initialize_from_ua_and_repr(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [(slice(0, 2), slice(0, 2))]

    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)
    ua = UncertainArray.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=2.0)

    aua.initialize_from_ua(ua)
    assert xp.allclose(aua.precision, ua.precision(raw=False))
    assert xp.allclose(aua.weighted_data, ua.data * ua.precision(raw=True))

    # repr sanity check
    rep = repr(aua)
    assert "AUA" in rep and "event_shape" in rep


@pytest.mark.parametrize("xp", backend_libs)
def test_initialize_from_ua_invalid_cases(xp):
    backend.set_backend(xp)
    event_shape = (2, 2)
    indices = [(slice(0, 2), slice(0, 2))]
    aua = AccumulativeUncertainArray(event_shape, indices)

    ua_batched = UncertainArray.zeros(event_shape, batch_size=3)
    with pytest.raises(ValueError):
        aua.initialize_from_ua(ua_batched)

    ua_shape_mismatch = UncertainArray.zeros((3, 3), batch_size=1)
    with pytest.raises(ValueError):
        aua.initialize_from_ua(ua_shape_mismatch)


@pytest.mark.parametrize("xp", backend_libs)
def test_scatter_mul_and_extract_patches(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [
        (slice(0, 2), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
    ]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    # UA with two patches
    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    aua.scatter_mul(ua)

    # extract_patches should return same UA back
    ua_out = aua.extract_patches()
    assert ua_out.batch_size == 2
    assert ua_out.event_shape == (2, 2)
    # Each patch should have mean==0, precision==1
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

    # UA with two patches, values=1, precision=1
    ua = UncertainArray.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    aua.scatter_mul(ua)

    # Overlap at (1,1) should accumulate
    total_prec = aua.precision[0, 1, 1]
    total_weighted = aua.weighted_data[0, 1, 1]
    assert xp.allclose(total_prec, 2.0)
    assert xp.allclose(total_weighted, 0.0)  


@pytest.mark.parametrize("xp", backend_libs)
def test_as_uncertain_array(xp):
    backend.set_backend(xp)
    event_shape = (2, 2)
    indices = [(slice(0, 2), slice(0, 2))]
    aua = AccumulativeUncertainArray(event_shape, indices, dtype=xp.complex64)

    ua = UncertainArray.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=2.0)
    aua.initialize_from_ua(ua)

    ua_out = aua.as_uncertain_array()
    assert ua_out.batch_size == 1
    assert ua_out.event_shape == (2, 2)
    assert xp.allclose(ua_out.data, xp.zeros_like(ua_out.data))
    assert xp.allclose(ua_out.precision(raw=False), xp.full((1, 2, 2), 2.0))
