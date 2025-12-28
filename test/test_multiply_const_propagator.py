import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.wave import Wave
from gpie.graph.propagator.multiply_const_propagator import MultiplyConstPropagator
from gpie.core.uncertain_array import UncertainArray as UA
from gpie.core.rng_utils import get_rng
from gpie.core.types import PrecisionMode, UnaryPropagatorPrecisionMode, get_real_dtype

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _make_wave(xp, shape=(2, 2), B=1, dtype=None):
    dtype = dtype or xp.complex64
    return Wave(event_shape=shape, batch_size=B, dtype=dtype)


def _make_prop_and_connect(xp, wave, const):
    prop = MultiplyConstPropagator(const=const)
    out = prop @ wave
    return prop, out


def _slice_batched(arr, block):
    if block is None:
        return arr
    return arr[block]


# ---------------------------------------------------------------------
# Core deterministic mapping tests (no EP correction)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
@pytest.mark.parametrize("const_val", [
    2.0,
    np.array([[2.0, 2.0], [2.0, 2.0]]),
    np.array([[[2.0, 2.0], [2.0, 2.0]]]),
])
def test_compute_forward_deterministic_array_precision(xp, const_val):
    """
    _compute_forward returns an ARRAY-precision UA by construction,
    except when precision_mode == ARRAY_TO_SCALAR (handled elsewhere).
    This test verifies the base deterministic transform:
        mu_out = mu_in * const
        prec_out = prec_in / (|const|^2 + eps)
    """
    backend.set_backend(xp)
    rng = get_rng(seed=0)

    shape = (2, 2)
    B = 1
    wave = _make_wave(xp, shape=shape, B=B)

    prop, out = _make_prop_and_connect(xp, wave, const_val)

    ua_in = UA.random(event_shape=shape, batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[wave] = ua_in

    # choose a mode that does NOT trigger ARRAY_TO_SCALAR special path
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    msg_out = prop._compute_forward({"input": ua_in}, block=None)

    const = prop.const  # already (B, *shape)
    abs_sq = xp.abs(const) ** 2
    eps = xp.array(prop._eps, dtype=get_real_dtype(prop.const_dtype))

    expected_mu = ua_in.data * const
    expected_prec = ua_in.precision(raw=True) / (abs_sq + eps)

    assert msg_out.event_shape == shape
    assert msg_out.batch_size == B
    assert msg_out.precision_mode == PrecisionMode.ARRAY
    assert xp.allclose(msg_out.data, expected_mu)
    assert xp.allclose(msg_out.precision(raw=True), expected_prec)


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_backward_deterministic_array_precision(xp):
    """
    _compute_backward returns an ARRAY-precision UA by construction,
    except when precision_mode == SCALAR_TO_ARRAY (handled elsewhere).
    This test verifies the deterministic adjoint-like mapping:
        mu_in = mu_out * conj(const) / (|const|^2 + eps)
        prec_in = prec_out * |const|^2
    """
    backend.set_backend(xp)
    rng = get_rng(seed=1)

    shape = (2, 2)
    B = 1
    wave = _make_wave(xp, shape=shape, B=B)

    const = xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=xp.complex64)
    prop, out = _make_prop_and_connect(xp, wave, const)

    ua_out = UA.random(event_shape=shape, batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)

    # choose a mode that does NOT trigger SCALAR_TO_ARRAY special path
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)

    msg_in = prop._compute_backward(ua_out, exclude="input", block=None)

    const_full = prop.const
    const_conj = xp.conj(const_full)
    abs_sq = xp.abs(const_full) ** 2
    eps = xp.array(prop._eps, dtype=get_real_dtype(prop.const_dtype))

    expected_mu = ua_out.data * const_conj / (abs_sq + eps)
    expected_prec = ua_out.precision(raw=True) * abs_sq

    assert msg_in.event_shape == shape
    assert msg_in.batch_size == B
    assert msg_in.precision_mode == PrecisionMode.ARRAY
    assert xp.allclose(msg_in.data, expected_mu)
    assert xp.allclose(msg_in.precision(raw=True), expected_prec)


# ---------------------------------------------------------------------
# EP correction paths
#   - forward: ARRAY_TO_SCALAR
#   - backward: SCALAR_TO_ARRAY
# ---------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_compute_forward_array_to_scalar_initial_vs_steady(xp):
    """
    For ARRAY_TO_SCALAR, _compute_forward behaves differently:
      - initial (output_message is None): return ua.as_scalar_precision()
      - steady-state: q = (ua * out_msg.as_array_precision()).as_scalar_precision()
                      return q / out_msg
    """
    backend.set_backend(xp)
    rng = get_rng(seed=2)

    shape = (2, 2)
    B = 1
    wave = _make_wave(xp, shape=shape, B=B)

    const = 2.0
    prop, out = _make_prop_and_connect(xp, wave, const)

    ua_in = UA.random(event_shape=shape, batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[wave] = ua_in

    prop._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR)

    # --- initial ---
    assert prop.output_message is None
    msg_init = prop._compute_forward({"input": ua_in}, block=None)
    assert msg_init.precision_mode == PrecisionMode.SCALAR

    # --- steady-state ---
    # provide a scalar output message to enable EP correction
    out_msg = UA.random(event_shape=shape, batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.receive_message(out, out_msg)  # sets output_message

    msg_steady = prop._compute_forward({"input": ua_in}, block=None)
    assert msg_steady.precision_mode == PrecisionMode.SCALAR

    # sanity: steady should not be wildly different in magnitude
    assert msg_steady.data.shape == msg_init.data.shape


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_backward_scalar_to_array_initial_vs_steady(xp):
    """
    For SCALAR_TO_ARRAY, _compute_backward behaves differently:
      - if input message exists: q = (ua * in_msg.as_array_precision()).as_scalar_precision()
                                return q / in_msg
      - if input message missing: return ua.as_scalar_precision()
    """
    backend.set_backend(xp)
    rng = get_rng(seed=3)

    shape = (2, 2)
    B = 1
    wave = _make_wave(xp, shape=shape, B=B)

    const = xp.ones(shape, dtype=xp.complex64) * 2.0
    prop, out = _make_prop_and_connect(xp, wave, const)

    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    ua_out = UA.random(event_shape=shape, batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)

    # --- initial: no input message cached ---
    assert prop.input_messages.get(wave) is None
    msg_init = prop._compute_backward(ua_out, exclude="input", block=None)
    assert msg_init.precision_mode == PrecisionMode.SCALAR

    # --- steady-state: provide scalar input message to enable EP correction ---
    ua_in = UA.random(event_shape=shape, batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[wave] = ua_in

    msg_steady = prop._compute_backward(ua_out, exclude="input", block=None)
    assert msg_steady.precision_mode == PrecisionMode.SCALAR


# ---------------------------------------------------------------------
# __matmul__ responsibilities
# ---------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_matmul_sets_batch_and_broadcasts_const(xp):
    """
    __matmul__ must:
      - set self.event_shape, self.batch_size
      - broadcast const to (B, *event_shape)
      - pick dtype via get_lower_precision_dtype
    """
    backend.set_backend(xp)

    shape = (2, 2)
    B = 3
    wave = _make_wave(xp, shape=shape, B=B, dtype=xp.complex64)

    const = xp.array([2.0], dtype=xp.complex128)  # higher precision
    prop = MultiplyConstPropagator(const=const)
    out = prop @ wave

    assert prop.batch_size == B
    assert prop.event_shape == shape
    assert prop.const.shape == (B, *shape)
    assert out.batch_size == B
    assert out.event_shape == shape
    assert out.dtype == xp.complex64  # lowered

    # broadcasting error
    bad_const = xp.ones((3, 3), dtype=xp.complex64)
    prop_bad = MultiplyConstPropagator(bad_const)
    with pytest.raises(ValueError):
        _ = prop_bad @ wave


# ---------------------------------------------------------------------
# Block-wise behavior (slice correctness)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_compute_forward_block_slicing(xp):
    """
    When block is provided, _compute_forward must:
      - slice input message
      - slice const fields consistently
      - return UA with batch_size == block_size
    """
    backend.set_backend(xp)
    rng = get_rng(seed=4)

    shape = (2, 2)
    B = 4
    wave = _make_wave(xp, shape=shape, B=B)

    const = xp.ones((B, *shape), dtype=xp.complex64) * 2.0
    prop, out = _make_prop_and_connect(xp, wave, const)

    ua_in = UA.random(event_shape=shape, batch_size=B, dtype=xp.complex64, rng=rng, scalar_precision=True)
    prop.input_messages[wave] = ua_in
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)

    blk = slice(1, 3)
    msg_blk = prop._compute_forward({"input": ua_in}, block=blk)

    assert msg_blk.batch_size == 2
    assert msg_blk.event_shape == shape

    # deterministic check on that block
    const_blk = prop.const[blk]
    abs_sq_blk = xp.abs(const_blk) ** 2
    eps = xp.array(prop._eps, dtype=get_real_dtype(prop.const_dtype))

    expected_mu = ua_in.data[blk] * const_blk
    expected_prec = ua_in.precision(raw=True)[blk] / (abs_sq_blk + eps)

    assert xp.allclose(msg_blk.data, expected_mu)
    assert xp.allclose(msg_blk.precision(raw=True), expected_prec)


# ---------------------------------------------------------------------
# to_backend and sampling
# ---------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_to_backend_transfer(xp):
    backend.set_backend(xp)

    const = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.complex128)
    prop = MultiplyConstPropagator(const=const)

    new_backend = cp if xp is np and has_cupy else np
    backend.set_backend(new_backend)
    prop.to_backend()

    assert isinstance(prop.const, new_backend.ndarray)
    assert isinstance(prop.const_conj, new_backend.ndarray)
    assert isinstance(prop.const_abs_sq, new_backend.ndarray)
    assert isinstance(prop._eps, new_backend.ndarray)


@pytest.mark.parametrize("xp", backend_libs)
def test_get_sample_for_output(xp):
    backend.set_backend(xp)

    shape = (2, 2)
    B = 2
    wave = _make_wave(xp, shape=shape, B=B, dtype=xp.complex64)
    sample = xp.ones((B, *shape), dtype=xp.complex64)
    wave.set_sample(sample)

    prop, out = _make_prop_and_connect(xp, wave, const=2.0)
    y = prop.get_sample_for_output()
    assert xp.allclose(y, sample * 2.0)


# ---------------------------------------------------------------------
# precision mode / getters
# ---------------------------------------------------------------------
def test_set_precision_mode_conflicts_and_invalid():
    backend.set_backend(np)
    prop = MultiplyConstPropagator(2.0)

    with pytest.raises(ValueError):
        prop._set_precision_mode("invalid")

    prop._set_precision_mode(UnaryPropagatorPrecisionMode.ARRAY)
    with pytest.raises(ValueError):
        prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)


def test_set_precision_mode_forward_and_backward():
    backend.set_backend(np)
    shape = (2, 2)
    B = 1

    # forward: input SCALAR -> SCALAR_TO_ARRAY
    wave = Wave(event_shape=shape, batch_size=B, dtype=np.complex64)
    out = MultiplyConstPropagator(2.0) @ wave
    prop = out.parent
    wave._set_precision_mode(PrecisionMode.SCALAR)
    prop.set_precision_mode_forward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY

    # backward: output SCALAR -> ARRAY_TO_SCALAR
    wave = Wave(event_shape=shape, batch_size=B, dtype=np.complex64)
    out = MultiplyConstPropagator(2.0) @ wave
    prop = out.parent
    prop.output._set_precision_mode(PrecisionMode.SCALAR)
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR

    # backward: input SCALAR & output ARRAY -> SCALAR_TO_ARRAY
    wave = Wave(event_shape=shape, batch_size=B, dtype=np.complex64)
    out = MultiplyConstPropagator(2.0) @ wave
    prop = out.parent
    wave._set_precision_mode(PrecisionMode.SCALAR)
    prop.output._set_precision_mode(PrecisionMode.ARRAY)
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY

    # backward: both ARRAY -> ARRAY
    wave = Wave(event_shape=shape, batch_size=B, dtype=np.complex64)
    out = MultiplyConstPropagator(2.0) @ wave
    prop = out.parent
    wave._set_precision_mode(PrecisionMode.ARRAY)
    prop.output._set_precision_mode(PrecisionMode.ARRAY)
    prop.set_precision_mode_backward()
    assert prop.precision_mode_enum == UnaryPropagatorPrecisionMode.ARRAY


def test_get_input_output_precision_modes():
    backend.set_backend(np)
    wave = Wave(event_shape=(2, 2), batch_size=1, dtype=np.complex64)
    out = MultiplyConstPropagator(2.0) @ wave
    prop = out.parent

    # none
    assert prop.get_input_precision_mode(wave) is None
    assert prop.get_output_precision_mode() is None

    # SCALAR_TO_ARRAY
    prop._set_precision_mode(UnaryPropagatorPrecisionMode.SCALAR_TO_ARRAY)
    assert prop.get_input_precision_mode(wave) == PrecisionMode.SCALAR
    assert prop.get_output_precision_mode() == PrecisionMode.ARRAY

    # ARRAY_TO_SCALAR
    wave = Wave(event_shape=(2, 2), batch_size=1, dtype=np.complex64)
    out = MultiplyConstPropagator(2.0) @ wave
    prop = out.parent
    prop._precision_mode = UnaryPropagatorPrecisionMode.ARRAY_TO_SCALAR
    assert prop.get_input_precision_mode(wave) == PrecisionMode.ARRAY
    assert prop.get_output_precision_mode() == PrecisionMode.SCALAR


# ---------------------------------------------------------------------
# repr (new style)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("xp", backend_libs)
def test_repr_contains_core_fields(xp):
    backend.set_backend(xp)

    prop = MultiplyConstPropagator(2.0)
    # before matmul: generation/batch may be default
    r0 = repr(prop)
    assert "MultiplyConst" in r0 or "MultiplyConstProp" in r0

    wave = Wave(event_shape=(2, 2), batch_size=3, dtype=xp.complex64)
    _ = prop @ wave
    r1 = repr(prop)

    # should contain at least mode/batch/event_shape under the new repr style
    assert "batch" in r1

@pytest.mark.parametrize("xp", backend_libs)
def test_multiply_const_block_forward_matches_full(xp):
    backend.set_backend(xp)

    B = 3
    H, W = 2, 2
    const = xp.arange(B * H * W).reshape(B, H, W).astype(xp.complex64)

    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    prop = MultiplyConstPropagator(const)
    _ = prop @ wave

    ua = UA.zeros((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    for b in range(B):
        ua.data[b] = b + 1

    full = prop._compute_forward({"input": ua})

    block = slice(1, 3)
    blk = prop._compute_forward({"input": ua}, block=block)
    expected = full.extract_block(block)

    assert xp.allclose(blk.data, expected.data)
    assert xp.allclose(
        blk.precision(raw=False),
        expected.precision(raw=False),
    )

@pytest.mark.parametrize("xp", backend_libs)
def test_multiply_const_block_backward_matches_full(xp):
    backend.set_backend(xp)

    B = 3
    H, W = 2, 2
    const = xp.ones((B, H, W), dtype=xp.complex64)

    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    prop = MultiplyConstPropagator(const)
    _ = prop @ wave

    ua = UA.random((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    out = prop._compute_forward({"input": ua})
    full_back = prop._compute_backward(out, exclude="input")

    block = slice(0, 2)
    blk_back = prop._compute_backward(out, exclude="input", block=block)
    expected = full_back.extract_block(block)

    assert xp.allclose(blk_back.data, expected.data)
    assert xp.allclose(
        blk_back.precision(raw=False),
        expected.precision(raw=False),
    )

@pytest.mark.parametrize("xp", backend_libs)
def test_multiply_const_blockwise_reassembly_equals_full(xp):
    backend.set_backend(xp)

    B = 4
    H, W = 2, 2
    const = xp.arange(B * H * W).reshape(B, H, W).astype(xp.complex64)

    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    prop = MultiplyConstPropagator(const)
    _ = prop @ wave

    ua = UA.zeros((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    for b in range(B):
        ua.data[b] = b + 1

    full = prop._compute_forward({"input": ua})

    re_data = xp.zeros_like(full.data)
    re_prec = xp.zeros_like(full.precision(raw=False))

    for b in range(B):
        blk = slice(b, b + 1)
        part = prop._compute_forward({"input": ua}, block=blk)
        re_data[b] = part.data[0]
        re_prec[b] = part.precision(raw=False)[0]

    assert xp.allclose(re_data, full.data)
    assert xp.allclose(re_prec, full.precision(raw=False))

@pytest.mark.parametrize("xp", backend_libs)
def test_multiply_const_batch_independence(xp):
    backend.set_backend(xp)

    B = 3
    H, W = 2, 2
    const = xp.ones((B, H, W), dtype=xp.complex64)

    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    prop = MultiplyConstPropagator(const)
    _ = prop @ wave

    ua = UA.zeros((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    ua.data[0] = 1.0
    ua.data[1] = 10.0
    ua.data[2] = 100.0

    out = prop._compute_forward({"input": ua})

    assert not xp.allclose(out.data[0], out.data[1])
    assert not xp.allclose(out.data[1], out.data[2])
