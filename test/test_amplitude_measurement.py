import importlib.util
import pytest
import numpy as np

from gpie.core import backend
from gpie.graph.measurement.amplitude_measurement import AmplitudeMeasurement
from gpie.graph.wave import Wave
from gpie.core.uncertain_array import UncertainArray
from gpie.core.rng_utils import get_rng
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
def test_dtype_inference_from_input_dtype(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2) << wave
    assert m.input_dtype == xp.dtype(xp.complex64)
    assert m.observed_dtype == xp.dtype(xp.float32)


@pytest.mark.parametrize("xp", backend_libs)
def test_generate_and_update_observed(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex128)
    wave.set_sample(xp.ones((1, 2, 2), dtype=xp.complex128))
    m = AmplitudeMeasurement(var=1e-3) << wave

    rng = get_rng(seed=42)
    m._generate_sample(rng)
    assert m.get_sample() is not None

    m.update_observed_from_sample()
    assert isinstance(m.observed, UncertainArray)
    assert m.observed.batch_shape == (1,)
    assert m.observed.event_shape == (2, 2)
    assert np.issubdtype(m.observed.dtype, np.floating)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_with_mask_scalar_precision(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    wave.set_sample(xp.ones((1, 2, 2), dtype=xp.complex64))
    m = AmplitudeMeasurement(var=0.5, precision_mode="scalar") << wave

    data = xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32).reshape(1, 2, 2)
    mask = xp.array([[True, False], [True, False]], dtype=bool).reshape(1, 2, 2)
    m.set_observed(data, mask=mask)
    assert isinstance(m.observed, UncertainArray)
    assert m.observed.batch_shape == (1,)
    assert m.observed.event_shape == (2, 2)
    assert np.allclose(m.observed.precision(raw=True)[mask], 2.0)
    assert np.allclose(m.observed.precision(raw=True)[~mask], 0.0)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_invalid_dtype_raises(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,), dtype=xp.complex64)
    meas = AmplitudeMeasurement(var=0.1) << wave

    obs = xp.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=xp.complex64)
    with pytest.raises(TypeError):
        meas.set_observed(obs)


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_message_and_shapes(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    wave.set_sample(xp.ones((1, 2, 2), dtype=xp.complex64))
    
    m = AmplitudeMeasurement(var=1e-2, precision_mode="scalar") << wave

    m._generate_sample(get_rng(seed=0))
    m.update_observed_from_sample()

    incoming = UncertainArray(wave.get_sample(), dtype=xp.complex64, precision=1.0, batched=True)
    msg = m._compute_message(incoming)
    assert isinstance(msg, UncertainArray)
    assert msg.batch_shape == (1,)
    assert msg.event_shape == (2, 2)
    assert np.issubdtype(msg.dtype, xp.complexfloating)

@pytest.mark.parametrize("xp", backend_libs)
def test_compute_belief_shapes_and_type(xp):
    """compute_belief(incoming) should return a UncertainArray with correct batch/event shapes and dtype."""
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2, precision_mode="scalar") << wave

    # Observed amplitude (with batch dim)
    y = xp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=xp.float32)
    m.set_observed(y)

    # Incoming complex Gaussian message (with batch dim)
    z_mean = xp.array([[[1.0+0.0j, 2.0+0.5j], [3.0-0.5j, 4.0+1.0j]]], dtype=xp.complex64)
    ua_in = UncertainArray(z_mean, dtype=xp.complex64, precision=1.0, batched=True)

    belief = m.compute_belief(ua_in)
    assert isinstance(belief, UncertainArray)
    assert belief.batch_shape == (1,)
    assert belief.event_shape == (2, 2)
    assert np.issubdtype(belief.dtype, xp.complexfloating)


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_fitness_matches_manual_no_mask(xp):
    """compute_fitness() should equal mean( gamma * (|mu_belief| - y)^2 ) without mask."""
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2, precision_mode="array") << wave

    # Observed amplitude and precision
    y = xp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=xp.float32)
    gamma = xp.ones_like(y, dtype=xp.float32) * 5.0  # precision=5 everywhere
    m.set_observed(y, precision=gamma)

    # Incoming message
    z_mean = xp.array([[[0.9+0.0j, 2.1+0.0j], [3.2+0.0j, 3.7+0.0j]]], dtype=xp.complex64)
    ua_in = UncertainArray(z_mean, dtype=xp.complex64, precision=2.0, batched=True)

    # Compute fitness
    _ = m.compute_belief(ua_in)  # populate m.belief
    f = m.compute_fitness()

    # Manual expectation
    xp_mod = xp
    mu_abs = xp_mod.abs(m.belief.data)
    diff2 = (mu_abs - y) ** 2
    expected = xp_mod.mean(gamma * diff2)
    assert np.allclose(f, float(expected), atol=1e-6)


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_fitness_respects_mask_via_zero_precision(xp):
    """Masked elements (precision=0) must not affect the mean(gamma * diff2)."""
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2, precision_mode="array", with_mask = True) << wave
    m._set_precision_mode("array")

    # Observed amplitude with mask: right column masked out
    y = xp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=xp.float32)
    mask = xp.array([[[1, 0], [1, 0]]], dtype=bool)
    m.set_observed(y, mask=mask)

    # Incoming message
    z_mean = xp.array([[[1.0+0j, 20.0+0j], [3.0+0j, 40.0+0j]]], dtype=xp.complex64)
    ua_in = UncertainArray(z_mean, dtype=xp.complex64, precision=1.0, batched=True)

    # Fitness
    _ = m.compute_belief(ua_in)
    f = m.compute_fitness()

    # Check that precision is zero on masked elements
    gamma = m.observed.precision(raw=True)
    assert xp.all(gamma[~mask] == 0.0)

    # Expected: mean over all entries of gamma*diff2 (masked gamma=0 nulls contribution)
    mu_abs = xp.abs(m.belief.data)
    diff2 = (mu_abs - y) ** 2
    expected = xp.mean(gamma * diff2)
    assert np.isclose(f, float(expected), atol=1e-6)


@pytest.mark.parametrize("xp", backend_libs)
def test_message_damping_applied_on_second_call(xp):
    """_compute_message should apply damping on the second call when damping>0."""
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2, precision_mode="scalar", damping=0.5) << wave

    # Observed amplitude
    y = xp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=xp.float32)
    m.set_observed(y)

    # Incoming (keep it fixed across calls)
    z_mean = xp.array([[[0.5+0j, 2.5+0j], [2.5+0j, 4.5+0j]]], dtype=xp.complex64)
    ua_in = UncertainArray(z_mean, dtype=xp.complex64, precision=1.0, batched=True)

    # First call: establishes old_msg
    msg1 = m._compute_message(ua_in)

    # Build the "full_msg" that _compute_message would produce before damping on second call
    # We need to recompute belief and full_msg exactly as the method does.
    belief2 = m.compute_belief(ua_in)
    full_msg2 = belief2 / ua_in  # before masking/damping

    # Second call: with damping
    msg2 = m._compute_message(ua_in)

    # msg2 should be damped between full_msg2 and msg1 (since old_msg=msg1, alpha=damping)
    # UncertainArray.damp_with is typically linear in data/precision; we check data field here.
    alpha = m.damping
    expected_data = (1 - alpha) * full_msg2.data + alpha * msg1.data
    assert np.allclose(msg2.data, expected_data, atol=1e-6)

    # Shapes remain consistent
    assert msg2.batch_shape == (1,)
    assert msg2.event_shape == (2, 2)


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_message_applies_mask_branch(xp):
    """_compute_message should take the masked branch when _mask is set."""
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-3, with_mask = True) << wave

    data = xp.ones((1, 2, 2), dtype=xp.float32)
    mask = xp.array([[[1, 0], [1, 0]]], dtype=bool)
    m.set_observed(data, mask=mask)

    incoming = UncertainArray(
        xp.ones((1, 2, 2), dtype=xp.complex64),
        dtype=xp.complex64,
        precision=1.0,
        batched=True,
    )
    msg = m._compute_message(incoming.as_array_precision())

    # Masked-out positions must be zero
    assert xp.all(msg.data[~mask] == 0)
    assert xp.all(msg.precision(raw=True)[~mask] == 0)

@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_before_connect_raises(xp):
    """set_observed must raise RuntimeError when called before << connection."""
    backend.set_backend(xp)
    m = AmplitudeMeasurement(var=1e-3)
    data = xp.ones((1, 1, 1), dtype=xp.float32)
    with pytest.raises(RuntimeError):
        m.set_observed(data)


@pytest.mark.parametrize("xp", backend_libs)
def test_set_observed_and_set_sample_shape_mismatch_raise(xp):
    """set_observed and set_sample should raise on shape mismatch."""
    backend.set_backend(xp)
    wave = Wave(event_shape=(2, 2), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-3) << wave

    # Wrong shape for observed (missing batch dim)
    bad_obs = xp.ones((2, 2), dtype=xp.float32)
    with pytest.raises(ValueError):
        m.set_observed(bad_obs)

    # Wrong shape for sample
    sample = xp.ones((2, 2), dtype=xp.complex64)  # no batch dim
    with pytest.raises(ValueError):
        m.set_sample(sample)


@pytest.mark.parametrize("xp", backend_libs)
def test_compute_message_block_equivalence(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(4,4), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2, precision_mode="scalar") << wave

    # observed
    y = xp.abs(xp.arange(16, dtype=xp.float32).reshape(1,4,4) + 1)
    m.set_observed(y)

    # incoming
    z = xp.ones((1,4,4), dtype=xp.complex64)
    incoming = UncertainArray(z, dtype=xp.complex64, precision=1.0, batched=True)

    # full message
    full_msg = m._compute_message(incoming, block=None)

    # block messages
    B = incoming.batch_size
    block = slice(0,1)  # only first element
    msg_blk = m._compute_message(incoming, block)

    # Compare block part
    assert np.allclose(msg_blk.data, full_msg.data[block], atol=1e-6)
    assert np.allclose(msg_blk.precision(raw=True),
                       full_msg.precision(raw=True)[block], atol=1e-6)

@pytest.mark.parametrize("xp", backend_libs)
def test_block_damping_applied(xp):
    backend.set_backend(xp)
    wave = Wave(event_shape=(2,2), dtype=xp.complex64)
    m = AmplitudeMeasurement(var=1e-2, precision_mode="scalar", damping=0.5) << wave

    # Observed
    y = xp.ones((1,2,2), dtype=xp.float32)
    m.set_observed(y)

    # incoming must match wave.event_shape!
    incoming = UncertainArray.zeros(
        event_shape=(2,2), batch_size=1,
        dtype=xp.complex64, precision=1.0
    )
    m.input_messages[wave] = incoming

    # First call creates last_backward_messages
    m.backward(block=None)
    prev_msg = m.last_backward_messages[wave]

    # Modify incoming slightly to force different message
    incoming2 = UncertainArray(
        2 * xp.ones((1,2,2), dtype=xp.complex64),
        dtype=xp.complex64, precision=1.0, batched=True,
    )
    m.input_messages[wave] = incoming2  # replace incoming

    # block damping call
    msg_blk = m._compute_message(incoming2, block=None)

    # Expected (damping applied)
    raw_msg = m.compute_belief(incoming2) / incoming2
    expected = 0.5 * prev_msg.data + 0.5 * raw_msg.data

    assert np.allclose(msg_blk.data, expected, atol=1e-6)

@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_compute_belief_block_equals_full(xp):
    backend.set_backend(xp)

    B = 3
    H, W = 4, 4

    meas = AmplitudeMeasurement(var=1e-4, damping=0.0)
    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    meas.add_input("input", wave)
    meas.batch_size = B

    incoming = UncertainArray.random((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    observed = UncertainArray.random((H, W), batch_size=B, dtype=xp.float32, precision=1.0)

    # Full
    full = meas.compute_belief(incoming, observed=observed, block=None)

    # Block-wise
    re_data = xp.zeros_like(full.data)
    re_prec = xp.zeros_like(full.precision(raw=False))

    for b in range(B):
        blk = slice(b, b + 1)
        part = meas.compute_belief(
            incoming.extract_block(blk),
            observed=observed.extract_block(blk),
            block=blk,
        )
        re_data[b] = part.data[0]
        re_prec[b] = part.precision(raw=False)[0]

    assert xp.allclose(re_data, full.data)
    assert xp.allclose(re_prec, full.precision(raw=False))

@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_compute_message_blockwise_equals_full(xp):
    backend.set_backend(xp)

    B = 3
    H, W = 4, 4

    meas = AmplitudeMeasurement(var=1e-4, damping=0.0)
    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    meas.add_input("input", wave)
    meas.batch_size = B
    meas._set_precision_mode(PrecisionMode.SCALAR)
    incoming = UncertainArray.random((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    meas.input_messages[wave] = incoming

    meas.observed = UncertainArray.random((H, W), batch_size=B, dtype=xp.float32, precision=1.0)

    full = meas._compute_message(incoming, block=None)

    re_data = xp.zeros_like(full.data)
    re_prec = xp.zeros_like(full.precision(raw=False))

    for b in range(B):
        blk = slice(b, b + 1)
        part = meas._compute_message(incoming, block=blk)
        re_data[b] = part.data[0]
        re_prec[b] = part.precision(raw=False)[0]

    assert xp.allclose(re_data, full.data)
    assert xp.allclose(re_prec, full.precision(raw=False))


@pytest.mark.parametrize("xp", backend_libs)
def test_amplitude_batch_independence(xp):
    backend.set_backend(xp)

    B = 3
    H, W = 4, 4

    meas = AmplitudeMeasurement(var=1e-4, damping=0.0)
    wave = Wave(event_shape=(H, W), batch_size=B, dtype=xp.complex64)
    meas.add_input("input", wave)
    meas.input = wave
    meas.batch_size = B
    meas._set_precision_mode(PrecisionMode.SCALAR)
    incoming = UncertainArray.zeros((H, W), batch_size=B, dtype=xp.complex64, precision=1.0)
    incoming.data[0] = 1.0
    incoming.data[1] = 10.0
    incoming.data[2] = 100.0

    meas.input_messages[wave] = incoming
    meas.observed = UncertainArray.zeros((H, W), batch_size=B, dtype=xp.float32, precision=1.0)
    meas.observed.data[:] = 1.0

    out = meas._compute_message(incoming)

    assert not xp.allclose(out.data[0], out.data[1])
    assert not xp.allclose(out.data[1], out.data[2])
