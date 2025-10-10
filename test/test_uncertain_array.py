import pytest
import numpy as np
import importlib.util
import warnings

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode

# Optional CuPy
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


@pytest.mark.parametrize("xp", backend_libs)
def test_astype_complex_to_real_with_warning(xp):
    backend.set_backend(xp)
    # complex UA with nonzero imaginary part
    data = xp.array([[1+1j, 2+2j]], dtype=xp.complex64)
    ua = UncertainArray(data, dtype=xp.complex64, precision=2.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ua_real = ua.astype(xp.float32)
        # warning should be raised
        assert any("discard imaginary part" in str(wi.message) for wi in w)

    # dtype changed to real
    assert ua_real.is_real()
    # precision doubled
    assert np.allclose(ua_real.precision(raw=False), 4.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_astype_complex_to_real_no_warning_if_imag_zero(xp):
    backend.set_backend(xp)
    # complex UA with zero imaginary part
    data = xp.array([[1+0j, 2+0j]], dtype=xp.complex64)
    ua = UncertainArray(data, dtype=xp.complex64, precision=3.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ua_real = ua.astype(xp.float32)
        # no warnings expected
        assert len(w) == 0

    # precision doubled
    assert np.allclose(ua_real.precision(raw=False), 6.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_real_property_from_complex(xp):
    backend.set_backend(xp)
    data = xp.array([[1+2j, 3+4j]], dtype=xp.complex64)
    ua = UncertainArray(data, dtype=xp.complex64, precision=5.0)

    ua_real = ua.real
    # dtype is real counterpart
    assert ua_real.is_real()
    # precision doubled
    assert np.allclose(ua_real.precision(raw=False), 10.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_real_property_when_already_real(xp):
    backend.set_backend(xp)
    data = xp.array([[1.0, 2.0]], dtype=xp.float32)
    ua = UncertainArray(data, dtype=xp.float32, precision=7.0)

    ua_real = ua.real
    # should return self, not a new object
    assert ua_real is ua


@pytest.mark.parametrize("xp", backend_libs)
def test_init_scalar_and_array_precision_vectorized(xp):
    backend.set_backend(xp)
    ua_scalar = UncertainArray.zeros(event_shape=(4, 4), batch_size=3, precision=2.0, scalar_precision=True)
    ua_array = UncertainArray.zeros(event_shape=(4, 4), batch_size=3, precision=2.0, scalar_precision=False)

    assert ua_scalar.batch_size == 3
    assert ua_scalar.event_shape == (4, 4)
    assert ua_scalar.precision_mode == PrecisionMode.SCALAR
    assert np.allclose(ua_scalar.precision(), 2.0)

    assert ua_array.precision_mode == PrecisionMode.ARRAY
    assert ua_array.precision().shape == (3, 4, 4)


@pytest.mark.parametrize("xp", backend_libs)
def test_mul_and_div_vectorized(xp):
    backend.set_backend(xp)
    ua1 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=1.0)
    ua2 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=2.0)
    ua_mul = ua1 * ua2
    ua_recovered = ua_mul / ua2

    assert np.allclose(ua1.data, ua_recovered.data, atol=1e-5)


@pytest.mark.parametrize("xp", backend_libs)
def test_damp_with_extremes(xp):
    backend.set_backend(xp)
    ua1 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=1.0)
    ua2 = UncertainArray.random(event_shape=(2, 2), batch_size=5, precision=10.0)

    ua_0 = ua1.damp_with(ua2, alpha=0.0)
    ua_1 = ua1.damp_with(ua2, alpha=1.0)

    assert np.allclose(ua_0.data, ua1.data)
    assert np.allclose(ua_1.data, ua2.data)


@pytest.mark.parametrize("xp", backend_libs)
def test_product_reduce_over_batch_preserves_precision_mode(xp):
    backend.set_backend(xp)

    # Case 1: scalar precision mode
    ua_scalar = UncertainArray.random(event_shape=(4, 4), batch_size=10, precision=2.0, scalar_precision=True)
    scalar_precision =  ua_scalar.precision(raw = True)
    assert scalar_precision.shape == (10,1,1)

    reduced_scalar = ua_scalar.product_reduce_over_batch()

    assert reduced_scalar.event_shape == (4, 4)
    assert xp.allclose(
        reduced_scalar.precision(),
        xp.sum(ua_scalar.precision(), axis=0),
        atol=1e-5
    )
    # Precision mode should remain scalar
    assert reduced_scalar.precision_mode == ua_scalar.precision_mode
    #reduced_precision = reduced_scalar.precision(True)
    #assert reduced_precision.shape == (1,1,1)

    # Case 2: array precision mode
    ua_array = UncertainArray.random(event_shape=(4, 4), batch_size=10, precision=2.0, scalar_precision=False)
    reduced_array = ua_array.product_reduce_over_batch()

    assert reduced_array.event_shape == (4, 4)
    assert xp.allclose(
        reduced_array.precision(),
        xp.sum(ua_array.precision(), axis=0),
        atol=1e-5
    )
    # Precision mode should remain array
    assert reduced_array.precision_mode == ua_array.precision_mode



@pytest.mark.parametrize("xp", backend_libs)
def test_as_precision_roundtrip_vectorized(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(3, 3), batch_size=4, precision=2.0, scalar_precision=True)

    assert ua.precision_mode == PrecisionMode.SCALAR
    ua_array = ua.as_array_precision()
    assert ua_array.precision_mode == PrecisionMode.ARRAY
    ua_back = ua_array.as_scalar_precision()
    assert ua_back.precision_mode == PrecisionMode.SCALAR


@pytest.mark.parametrize("xp", backend_libs)
def test_repr_contains_batch_info(xp):
    backend.set_backend(xp)
    ua = UncertainArray.zeros(event_shape=(8,), batch_size=5)
    r = repr(ua)
    assert "batch_size=5" in r
    assert "event_shape=(8,)" in r


@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_roundtrip(xp):
    import numpy as np

    if not has_cupy:
        pytest.skip("CuPy not installed")
    else:
        import cupy as cp

    backend.set_backend(np)
    ua = UncertainArray.zeros(event_shape=(2, 2), dtype=np.complex64, precision=1.0)

    backend.set_backend(cp)
    ua.to_backend()
    assert isinstance(ua.data, cp.ndarray)
    assert ua.dtype == cp.complex64



from gpie.core import fft
import itertools

try:
    import pyfftw
    has_pyfftw = True
except ImportError:
    has_pyfftw = False


@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("fft_backend", ["numpy", "fftw"] if has_pyfftw else ["numpy"])
def test_fft2_ifft2_centered_reconstruction(xp, fft_backend):
    backend.set_backend(xp)
    if fft_backend == "fftw" and xp.__name__ != "numpy":
        pytest.skip("FFTW backend requires NumPy")

    fft.set_fft_backend(fft_backend)

    ua = UncertainArray.random(
        event_shape=(32, 32),
        batch_size=4,
        dtype=xp.complex64,
        scalar_precision=False,
    )

    ua_hat = ua.fft2_centered()
    assert ua_hat.precision_mode == PrecisionMode.SCALAR

    ua_rec = ua_hat.ifft2_centered()
    assert ua_rec.precision_mode == PrecisionMode.SCALAR

    assert np.allclose(ua.data, ua_rec.data, atol=1e-5), f"FFT->IFFT failed for {fft_backend}, {xp.__name__}"

@pytest.mark.parametrize("xp", backend_libs)
def test_fork_basic_and_error(xp):
    backend.set_backend(xp)
    ua = UncertainArray.zeros(event_shape=(4, 4), batch_size=1, precision=2.0)
    ua4 = ua.fork(batch_size=4)
    assert ua4.batch_size == 4
    assert ua4.event_shape == (4, 4)
    # All copies should match original
    for i in range(4):
        assert np.allclose(ua4.data[i], ua.data[0])
        assert np.allclose(ua4.precision()[i], ua.precision()[0])
    # Error if batch_size != 1
    ua_multi = UncertainArray.zeros(event_shape=(4, 4), batch_size=2)
    with pytest.raises(ValueError):
        _ = ua_multi.fork(batch_size=3)


@pytest.mark.parametrize("xp", backend_libs)
def test_zero_pad_data_and_precision(xp):
    backend.set_backend(xp)
    ua = UncertainArray.zeros(event_shape=(4, 4), batch_size=1, precision=5.0)
    ua_padded = ua.zero_pad(((1, 1), (2, 2)))
    assert ua_padded.event_shape == (6, 8)
    # Original data region should remain zero, padded region also zero
    assert xp.allclose(ua_padded.data, 0.0)
    # Precision in pad region must be zero
    center_prec = ua_padded.precision()[0, 1:-1, 2:-2]
    pad_prec = ua_padded.precision()[0]
    assert xp.allclose(center_prec, 5.0)
    assert xp.all(pad_prec >= 0)
    assert xp.allclose(pad_prec[:, :2], 1e8)  # left pad zero


@pytest.mark.parametrize("xp", backend_libs)
def test_getitem_basic_and_error(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(8, 8), batch_size=1, precision=3.0)
    sub = ua[2:6, 2:6]
    assert sub.event_shape == (4, 4)
    assert sub.batch_size == 1
    # Error if batch_size != 1
    ua_multi = UncertainArray.random(event_shape=(8, 8), batch_size=2)
    with pytest.raises(ValueError):
        _ = ua_multi[2:6, 2:6]


@pytest.mark.parametrize("xp", backend_libs)
def test_extract_patches_basic_and_error(xp):
    backend.set_backend(xp)
    ua = UncertainArray.random(event_shape=(8, 8), batch_size=1, precision=1.0)
    patches = ua.extract_patches([
        (slice(0, 4), slice(0, 4)),
        (slice(4, 8), slice(4, 8))
    ])
    assert patches.batch_size == 2
    assert patches.event_shape == (4, 4)
    # Check that extracted patches match original UA data
    assert xp.allclose(patches.data[0], ua.data[0, 0:4, 0:4])
    assert xp.allclose(patches.data[1], ua.data[0, 4:8, 4:8])
    # Error if batch_size != 1
    ua_multi = UncertainArray.random(event_shape=(8, 8), batch_size=2)
    with pytest.raises(ValueError):
        _ = ua_multi.extract_patches([(slice(0, 4), slice(0, 4))])
