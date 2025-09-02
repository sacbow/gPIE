import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.binary_propagator import BinaryPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.types import PrecisionMode, BinaryPropagatorPrecisionMode as BPM

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


class DummyBinary(BinaryPropagator):
    """Concrete subclass for testing forward/backward message passing."""
    def _compute_forward(self, inputs):
        # Simple sum of means for test purposes
        return UA(inputs["a"].data + inputs["b"].data,
                  dtype=inputs["a"].dtype,
                  precision=inputs["a"].precision(raw=True))

    def _compute_backward(self, output, exclude: str):
        return UA(output.data, dtype=output.dtype, precision=output.precision(raw=True))


@pytest.mark.parametrize("xp", backend_libs)
def test_dtype_and_shape_check(xp):
    """Test dtype resolution and shape mismatch in __matmul__."""
    backend.set_backend(xp)
    a = Wave(shape=(2, 2), dtype=xp.float32)
    b = Wave(shape=(2, 2), dtype=xp.complex64)

    # dtype should be lowered to complex64
    prop = DummyBinary()
    out = prop @ (a, b)
    assert out.dtype == xp.complex64

    # shape mismatch raises
    b_bad = Wave(shape=(3, 3), dtype=xp.complex64)
    prop_bad = DummyBinary()
    with pytest.raises(ValueError):
        _ = prop_bad @ (a, b_bad)

    # non-tuple inputs raise
    with pytest.raises(ValueError):
        _ = prop @ a  # not a tuple


@pytest.mark.parametrize("xp", backend_libs)
def test_precision_mode_forward(xp):
    """Test all combinations of precision modes in set_precision_mode_forward."""
    backend.set_backend(xp)

    # scalar-scalar
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    a._set_precision_mode("scalar")
    b._set_precision_mode("scalar")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.SCALAR

    # array-array
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    a._set_precision_mode("array")
    b._set_precision_mode("array")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.ARRAY

    # scalar-array
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    a._set_precision_mode("scalar")
    b._set_precision_mode("array")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.SCALAR_AND_ARRAY_TO_ARRAY

    # array-scalar
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    a._set_precision_mode("array")
    b._set_precision_mode("scalar")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.ARRAY_AND_SCALAR_TO_ARRAY

    # one unset, one array
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    b._set_precision_mode("array")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.ARRAY

    # one unset, one scalar (should leave unset)
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    b._set_precision_mode("scalar")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum is None


@pytest.mark.parametrize("xp", backend_libs)
def test_precision_mode_backward(xp):
    """Test backward precision mode resolution and error cases."""
    backend.set_backend(xp)

    # output array + both unset -> ARRAY
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    output._set_precision_mode("array")
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == BPM.ARRAY

    # output array + (a unset, b scalar) -> ARRAY_AND_SCALAR_TO_ARRAY
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    b._set_precision_mode("scalar")
    output._set_precision_mode("array")
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == BPM.ARRAY_AND_SCALAR_TO_ARRAY

    # output array + (a scalar, b unset) -> SCALAR_AND_ARRAY_TO_ARRAY
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    a._set_precision_mode("scalar")
    output._set_precision_mode("array")
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == BPM.SCALAR_AND_ARRAY_TO_ARRAY

    # inconsistent: output array + both scalar -> raises
    a._set_precision_mode("scalar")
    b._set_precision_mode("scalar")
    output._set_precision_mode("array")
    with pytest.raises(ValueError):
        prop.set_precision_mode_backward()


@pytest.mark.parametrize("xp", backend_libs)
def test_get_input_and_output_precision_mode(xp):
    """Test get_input_precision_mode and get_output_precision_mode logic."""
    backend.set_backend(xp)

    # Scalar mode
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.SCALAR)
    assert prop.get_output_precision_mode()=="scalar"
    assert prop.get_input_precision_mode(b)== "scalar"

    # Array mode
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.ARRAY)
    assert prop.get_output_precision_mode() == "array"
    assert prop.get_input_precision_mode(b) == "array"

    # Scalar+Array to Array
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
    assert prop.get_input_precision_mode(a) == "scalar"
    assert prop.get_input_precision_mode(b) == "array"

    # Array+Scalar to Array
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
    assert prop.get_input_precision_mode(a) == "array"
    assert prop.get_input_precision_mode(b) == "scalar"


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_and_backward_message_passing(xp):
    """Test forward/backward message passing works with dummy UA compute."""
    backend.set_backend(xp)
    a = Wave((2, 2), dtype=xp.complex64)
    b = Wave((2, 2), dtype=xp.complex64)
    prop = DummyBinary() @ (a, b)
    parent = prop.parent
    parent._set_precision_mode(BPM.SCALAR)

    rng = np.random.default_rng(0)
    ua_a = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision=True)
    ua_b = UA.random((2, 2), dtype=xp.complex64, rng=rng, scalar_precision=True)
    parent.receive_message(a, ua_a)
    parent.receive_message(b, ua_b)

    # Forward computes UA and sends to output
    parent.forward()
    assert isinstance(prop.parent_message, UA)


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_missing_messages_and_not_implemented(xp):
    """Test error cases for missing messages and abstract method enforcement."""
    backend.set_backend(xp)
    a = Wave((2, 2), dtype=xp.float32)
    b = Wave((2, 2), dtype=xp.float32)
    prop = BinaryPropagator()  # base class directly
    prop.add_input("a", a)
    prop.add_input("b", b)
    prop.output = Wave((2, 2), dtype=xp.float32)

    # forward with missing inputs raises
    with pytest.raises(RuntimeError):
        prop.forward()

    # backward with missing output raises
    with pytest.raises(RuntimeError):
        prop.backward()

    # abstract methods raise
    ua = UA.random((2, 2), dtype=xp.float32, rng=np.random.default_rng(), scalar_precision=True)
    with pytest.raises(NotImplementedError):
        prop._compute_forward({"a": ua, "b": ua})
    with pytest.raises(NotImplementedError):
        prop._compute_backward(ua, exclude="a")
