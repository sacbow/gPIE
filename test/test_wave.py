import importlib.util
import numpy as np
import pytest

from gpie.core import backend
from gpie.graph.wave import Wave
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
def test_wave_set_and_get_sample(xp):
    backend.set_backend(xp)
    w = Wave(shape=(4, 4), dtype=xp.complex128)
    sample = xp.ones((4, 4), dtype=xp.complex128)
    w.set_sample(sample)
    assert xp.allclose(w.get_sample(), sample)
    w.clear_sample()
    assert w.get_sample() is None


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_finalize_structure_scalar(xp):
    backend.set_backend(xp)
    w = Wave(shape=(4, 4), dtype=xp.complex128)
    w._set_precision_mode("scalar")
    w.children = [object(), object()]  # Dummy children
    w.finalize_structure()
    assert isinstance(w.child_messages_tensor, UncertainArrayTensor)
    assert w.child_messages_tensor.precision_mode == PrecisionMode.SCALAR


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_receive_message_dtype_casting(xp):
    backend.set_backend(xp)
    w = Wave(shape=(2, 2), dtype=xp.complex128)
    real_message = UncertainArray(xp.ones((2, 2), dtype=xp.float64), precision=1.0)
    # Acceptable: real → complex
    w.set_parent(object())
    w.receive_message(w.parent, real_message)
    assert w.parent_message.dtype == xp.complex128


@pytest.mark.parametrize("xp", backend_libs)
def test_wave_compute_belief_fuses_messages(xp):
    backend.set_backend(xp)
    w = Wave(shape=(2, 2), dtype=xp.complex128)
    w._set_precision_mode("scalar")
    w.children = [object(), object()]
    w.finalize_structure()

    # Dummy child messages (same as initialized ones)
    parent_msg = UncertainArray(xp.full((2, 2), 1.0), precision=2.0)
    w.set_parent(object())
    w.receive_message(w.parent, parent_msg)
    belief = w.compute_belief()
    assert isinstance(belief, UncertainArray)
    assert xp.allclose(belief.data.shape, (2, 2))

@pytest.mark.parametrize("xp", backend_libs)
def test_wave_to_backend_updates_dtype_and_storage(xp):
    backend.set_backend(xp)
    w = Wave(shape=(2, 2), dtype=xp.complex128)
    w._set_precision_mode("array")
    w.children = [object(), object()]
    w.finalize_structure()
    # 明示的に belief / parent_message もセット
    w.set_parent(object())
    parent_msg = UncertainArray(xp.full((2, 2), 1.0), precision=xp.ones((2, 2)))
    w.receive_message(w.parent, parent_msg)

    w.compute_belief()

    # CPUからGPUへ（またはその逆）移す
    new_backend = cp if xp is np else np
    backend.set_backend(new_backend)
    w.to_backend()

    assert isinstance(w.child_messages_tensor.data, new_backend.ndarray)
    assert isinstance(w.belief.data, new_backend.ndarray)
    assert w.dtype == new_backend.complex128
