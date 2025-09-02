import importlib.util
import numpy as np
import pytest

from gpie.core import backend
from gpie.core.uncertain_array import UncertainArray
from gpie.core.uncertain_array_tensor import UncertainArrayTensor
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
def test_from_list_and_to_list_roundtrip(xp):
    backend.set_backend(xp)
    ua_list = [
        UncertainArray(xp.ones((2, 2)) * i, precision=1.0 + i)
        for i in range(3)
    ]
    uat = UncertainArrayTensor.from_list(ua_list)
    out_list = uat.to_list()
    for ua_in, ua_out in zip(ua_list, out_list):
        assert np.allclose(ua_in.data, ua_out.data)
        assert np.allclose(ua_in.precision(), ua_out.precision())
        assert ua_in.precision_mode == ua_out.precision_mode


@pytest.mark.parametrize("xp", backend_libs)
def test_scalar_mode_combine_correctness(xp):
    backend.set_backend(xp)
    data = xp.array([
        [[1.0, 2.0],
         [3.0, 4.0]],
        [[5.0, 6.0],
         [7.0, 8.0]],
    ])
    precision = xp.array([1.0, 3.0])
    uat = UncertainArrayTensor(data, precision)
    fused = uat.combine()
    expected = (1.0 * data[0] + 3.0 * data[1]) / 4.0
    assert np.allclose(fused.data, expected)
    assert fused.precision_mode == PrecisionMode.SCALAR
    assert np.isclose(fused.precision(raw=True), 4.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_array_mode_combine_correctness(xp):
    backend.set_backend(xp)
    data = xp.array([
        [[1.0, 2.0],
         [3.0, 4.0]],
        [[5.0, 6.0],
         [7.0, 8.0]],
    ])
    precision = xp.array([
        [[1.0, 1.0],
         [1.0, 1.0]],
        [[3.0, 3.0],
         [3.0, 3.0]],
    ])
    uat = UncertainArrayTensor(data, precision)
    fused = uat.combine()
    expected = (1.0 * data[0] + 3.0 * data[1]) / 4.0
    assert np.allclose(fused.data, expected)
    assert fused.precision_mode == PrecisionMode.ARRAY
    assert np.allclose(fused.precision(), 4.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_setitem_and_getitem_behavior(xp):
    backend.set_backend(xp)
    ua1 = UncertainArray(xp.ones((2, 2)), precision=2.0)
    ua2 = UncertainArray(xp.full((2, 2), 3.0), precision=1.0)
    uat = UncertainArrayTensor.from_list([ua1, ua1])
    uat[1] = ua2
    out = uat[1]
    assert np.allclose(out.data, ua2.data)
    assert np.allclose(out.precision(), ua2.precision())


@pytest.mark.parametrize("xp", backend_libs)
def test_assert_compatible_raises_on_mismatch(xp):
    backend.set_backend(xp)

    ua1 = UncertainArray(xp.ones((2, 2), dtype=xp.complex128), precision=1.0)
    uat = UncertainArrayTensor.from_list([ua1, ua1])

    ua_wrong_shape = UncertainArray(xp.ones((3, 3), dtype=xp.complex128), precision=1.0)
    with pytest.raises(ValueError, match="Shape mismatch"):
        uat.assert_compatible(ua_wrong_shape)

    ua_wrong_dtype = UncertainArray(xp.ones((2, 2), dtype=xp.float64), precision=1.0)
    with pytest.raises(TypeError, match="Dtype mismatch"):
        uat.assert_compatible(ua_wrong_dtype)

    ua_wrong_prec = UncertainArray(xp.ones((2, 2), dtype=xp.complex128),
                                   precision=xp.full((2, 2), 1.0))
    with pytest.raises(ValueError, match="Precision mode mismatch"):
        uat.assert_compatible(ua_wrong_prec)

