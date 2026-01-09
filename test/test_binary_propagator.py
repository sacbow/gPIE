import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.core.rng_utils import get_rng
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
        return UA(inputs["a"].data + inputs["b"].data,
                  dtype=inputs["a"].dtype,
                  precision=inputs["a"].precision(raw=True))

    def _compute_backward(self, output, exclude: str):
        return UA(output.data, dtype=output.dtype, precision=output.precision(raw=True))


@pytest.mark.parametrize("xp", backend_libs)
def test_dtype_and_shape_check(xp):
    backend.set_backend(xp)
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.complex64)

    prop = DummyBinary()
    out = prop @ (a, b)
    assert out.dtype == xp.complex64

    b_bad = Wave(event_shape=(3, 3), dtype=xp.complex64)
    prop_bad = DummyBinary()
    with pytest.raises(ValueError):
        _ = prop_bad @ (a, b_bad)

    with pytest.raises(ValueError):
        _ = prop @ a  # not a tuple


@pytest.mark.parametrize("xp", backend_libs)
def test_precision_mode_forward(xp):
    backend.set_backend(xp)

    # scalar-scalar → no mode assigned
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    a._set_precision_mode("scalar")
    b._set_precision_mode("scalar")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.SCALAR

    # array-array → no mode assigned
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    a._set_precision_mode("array")
    b._set_precision_mode("array")
    prop = output.parent
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum is None

    # scalar-array
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    a._set_precision_mode("scalar")
    b._set_precision_mode("array")
    prop = output.parent
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.SCALAR_AND_ARRAY_TO_ARRAY

    # array-scalar
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    a._set_precision_mode("array")
    b._set_precision_mode("scalar")
    prop = output.parent
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == BPM.ARRAY_AND_SCALAR_TO_ARRAY

    # one unset, one array
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    b._set_precision_mode("array")
    prop = output.parent
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum is None

    # one unset, one scalar → no assignment
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    b._set_precision_mode("scalar")
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum is None


@pytest.mark.parametrize("xp", backend_libs)
def test_precision_mode_backward(xp):
    backend.set_backend(xp)

    # output=array + both unset → ARRAY
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    output._set_precision_mode("array")
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == BPM.ARRAY

    # output=array + (a unset, b scalar) → ARRAY_AND_SCALAR_TO_ARRAY
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    b._set_precision_mode("scalar")
    prop = output.parent
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == BPM.SCALAR

    # output=scalar + both inputs array → ARRAY_AND_ARRAY_TO_SCALAR
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    a._set_precision_mode("array")
    b._set_precision_mode("array")
    output._set_precision_mode("scalar")
    prop = output.parent
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == BPM.ARRAY_AND_ARRAY_TO_SCALAR

    # output=scalar + both inputs scalar → SCALAR
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    a._set_precision_mode("scalar")
    b._set_precision_mode("scalar")
    output._set_precision_mode("scalar")
    prop = output.parent
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == BPM.SCALAR


@pytest.mark.parametrize("xp", backend_libs)
def test_get_input_and_output_precision_mode(xp):
    backend.set_backend(xp)

    # SCALAR
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.SCALAR)
    output.set_precision_mode_forward()
    assert prop.get_output_precision_mode() == "scalar"
    a.set_precision_mode_backward()
    assert prop.get_input_precision_mode(a) == "scalar"

    # ARRAY
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.ARRAY)
    assert prop.get_output_precision_mode() == "array"
    assert prop.get_input_precision_mode(a) == "array"

    # ARRAY_AND_ARRAY_TO_SCALAR
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.ARRAY_AND_ARRAY_TO_SCALAR)
    assert prop.get_output_precision_mode() == "scalar"
    assert prop.get_input_precision_mode(a) == "array"
    assert prop.get_input_precision_mode(b) == "array"

    # MIXED: scalar/array to array
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.SCALAR_AND_ARRAY_TO_ARRAY)
    assert prop.get_input_precision_mode(a) == "scalar"
    assert prop.get_input_precision_mode(b) == "array"

    # MIXED: array/scalar to array
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    output = DummyBinary() @ (a, b)
    prop = output.parent
    prop._set_precision_mode(BPM.ARRAY_AND_SCALAR_TO_ARRAY)
    assert prop.get_input_precision_mode(a) == "array"
    assert prop.get_input_precision_mode(b) == "scalar"


@pytest.mark.parametrize("xp", backend_libs)
def test_forward_backward_missing_messages_and_not_implemented(xp):
    backend.set_backend(xp)
    a = Wave(event_shape=(2, 2), dtype=xp.float32)
    b = Wave(event_shape=(2, 2), dtype=xp.float32)
    prop = BinaryPropagator()
    prop.add_input("a", a)
    prop.add_input("b", b)
    prop.output = Wave(event_shape=(2, 2), dtype=xp.float32)

    with pytest.raises(RuntimeError):
        prop.forward()

    with pytest.raises(RuntimeError):
        prop.backward()

    ua = UA.random((2, 2), dtype=xp.float32, rng=get_rng(), scalar_precision=True)
    with pytest.raises(NotImplementedError):
        prop._compute_forward({"a": ua, "b": ua})
    with pytest.raises(NotImplementedError):
        prop._compute_backward(ua, exclude="a")
