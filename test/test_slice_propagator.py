import pytest
import numpy as np
import importlib.util

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.graph.propagator.slice_propagator import SlicePropagator

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


# ----------------------------------------------------------------------
# Initialization tests
# ----------------------------------------------------------------------

def test_slice_propagator_init_invalid_types():
    with pytest.raises(TypeError, match="tuple of slices or a list of tuples of slices"):
        _ = SlicePropagator("not a slice")

    with pytest.raises(TypeError, match="tuple of slices"):
        _ = SlicePropagator((1, 2))

    with pytest.raises(TypeError, match="tuple of slices"):
        _ = SlicePropagator([1, 2])

    with pytest.raises(TypeError, match="tuple of slices"):
        _ = SlicePropagator([(slice(0, 2), 1)])


def test_slice_propagator_init_shape_mismatch():
    with pytest.raises(ValueError, match="must produce patches of the same shape"):
        _ = SlicePropagator([
            (slice(0, 2), slice(0, 2)),
            (slice(0, 3), slice(0, 3)),
        ])


# ----------------------------------------------------------------------
# __matmul__ (graph construction)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_normal_case(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    indices = [(slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4))]
    prop = SlicePropagator(indices)

    out = prop @ wave
    assert out.batch_size == 2
    assert out.event_shape == (2, 2)
    assert out.dtype == wave.dtype
    assert out.parent is prop
    # output_product must be initialized
    assert prop.output_product is not None
    assert prop.output_product.event_shape == wave.event_shape


@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_batch_size_error(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=2, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 2), slice(0, 2))])

    with pytest.raises(ValueError, match="batch_size=1"):
        _ = prop @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_rank_mismatch(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 2),)])  # 1D slice, mismatch
    with pytest.raises(ValueError, match="Slice rank mismatch"):
        _ = prop @ wave


@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_out_of_range(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 5), slice(0, 4))])
    with pytest.raises(ValueError, match="out of range"):
        _ = prop @ wave


# ----------------------------------------------------------------------
# Forward and backward
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_forward_extracts_patches(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [(slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4))]
    wave = Wave(event_shape=event_shape, batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator(indices)
    _ = prop @ wave

    ua = UA.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0

    out = prop._compute_forward({"input": ua})
    assert out.batch_size == 2
    assert out.event_shape == (2, 2)
    assert xp.allclose(out.data, 1.0)
    assert xp.allclose(out.precision(raw=False), 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_backward_reconstructs_input(xp):
    backend.set_backend(xp)
    event_shape = (4, 4)
    indices = [(slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4))]
    wave = Wave(event_shape=event_shape, batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator(indices)
    _ = prop @ wave

    ua = UA.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 2.0
    prop._compute_forward({"input": ua})

    patch_ua = UA.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    patch_ua.data[...] = 3.0

    recon = prop._compute_backward(patch_ua, exclude="input")
    assert recon.batch_size == 1
    assert recon.event_shape == event_shape

    expected = xp.zeros(event_shape, dtype=xp.complex64)
    expected[0:2, 0:2] = 3.0
    expected[2:4, 2:4] = 3.0
    assert xp.allclose(recon.data[0], expected)


@pytest.mark.parametrize("xp", backend_libs)
def test_backward_with_overlap(xp):
    backend.set_backend(xp)
    event_shape = (3, 3)
    indices = [(slice(0, 2), slice(0, 2)), (slice(1, 3), slice(1, 3))]
    wave = Wave(event_shape=event_shape, batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator(indices)
    _ = prop @ wave

    ua = UA.zeros(event_shape, batch_size=1, dtype=xp.complex64, precision=1.0)
    ua.data[...] = 1.0
    prop._compute_forward({"input": ua})

    patch_ua = UA.zeros((2, 2), batch_size=2, dtype=xp.complex64, precision=1.0)
    patch_ua.data[...] = 1.0

    recon = prop._compute_backward(patch_ua, exclude="input")
    assert np.isclose(recon.precision(raw=False)[0, 1, 1], 2.0)
    assert np.isclose(recon.data[0, 1, 1], 1.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_error_cases(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4))])
    _ = prop @ wave

    ua_invalid = UA.zeros((4, 4), batch_size=2, dtype=xp.complex64, precision=1.0)
    with pytest.raises(ValueError):
        prop._compute_forward({"input": ua_invalid})

    patch_ua_invalid = UA.zeros((2, 2), batch_size=1, dtype=xp.complex64, precision=1.0)
    with pytest.raises(ValueError):
        prop._compute_backward(patch_ua_invalid, exclude="input")


# ----------------------------------------------------------------------
# get_sample_for_output
# ----------------------------------------------------------------------

@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output_normal(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    sample = xp.arange(16, dtype=xp.complex64).reshape(1, 4, 4)
    wave.set_sample(sample)

    indices = [(slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4))]
    prop = SlicePropagator(indices)
    _ = prop @ wave

    patches = prop.get_sample_for_output()
    assert patches.shape == (2, 2, 2)

    expected1 = sample[0, 0:2, 0:2]
    expected2 = sample[0, 2:4, 2:4]
    assert xp.allclose(patches[0], expected1)
    assert xp.allclose(patches[1], expected2)


@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output_without_sample(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4, 4), batch_size=1, dtype=xp.complex64)
    prop = SlicePropagator([(slice(0, 2), slice(0, 2))])
    _ = prop @ wave

    with pytest.raises(RuntimeError, match="Input sample not set"):
        _ = prop.get_sample_for_output()
