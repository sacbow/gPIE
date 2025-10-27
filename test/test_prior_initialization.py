# test/test_prior_initialization.py
import pytest
import numpy as np
from gpie.core.uncertain_array import UncertainArray
from gpie.core.types import PrecisionMode
from gpie.graph.prior.base import Prior


# ---- Dummy subclass for testing Prior initialization behavior ----
class DummyPrior(Prior):
    """Minimal Prior subclass for initialization testing."""
    called_get_sample = False

    def _compute_message(self, incoming):
        return incoming

    def get_sample_for_output(self, rng=None):
        DummyPrior.called_get_sample = True
        return np.ones((self.batch_size, *self.event_shape), dtype=self.dtype)


# ============================================================
# 1. set_manual_init() behavior
# ============================================================

def test_set_manual_init_auto_reshape_for_single_sample():
    """When user provides (event_shape,) array, it should reshape to (batch_size, *event_shape)."""
    p = DummyPrior(event_shape=(2, 2), batch_size=1)
    data = np.ones((2, 2))
    p.set_manual_init(data)
    ua = p._manual_init_msg
    assert isinstance(ua, UncertainArray)
    assert ua.data.shape == (1, 2, 2)
    assert np.allclose(ua.data, 1.0)
    assert np.allclose(ua.precision(), 1.0)


def test_set_manual_init_shape_mismatch_raises():
    """If shape does not match (batch_size, *event_shape), ValueError should be raised."""
    p = DummyPrior(event_shape=(3, 3), batch_size=2)
    bad = np.ones((3, 3))  # expected (2,3,3)
    with pytest.raises(ValueError, match="Manual init shape mismatch"):
        p.set_manual_init(bad)


def test_set_manual_init_with_custom_precision_scalar():
    """Custom scalar precision should be reflected in the UncertainArray."""
    p = DummyPrior(event_shape=(2, 2))
    arr = np.full((1, 2, 2), 7.0)
    p.set_manual_init(arr, precision=2.5)
    ua = p._manual_init_msg
    assert np.allclose(ua.data, 7.0)
    assert np.allclose(ua.precision(), 2.5)


# ============================================================
# 2. _get_initial_message() under different strategies
# ============================================================

@pytest.mark.parametrize("mode", [PrecisionMode.SCALAR, PrecisionMode.ARRAY])
def test_get_initial_message_manual_strategy(mode):
    """Manual init strategy should return manual UA and match precision mode."""
    p = DummyPrior(event_shape=(2, 2), precision_mode=mode)
    p.set_manual_init(np.full((1, 2, 2), 9.0))
    p.set_init_strategy("manual")
    msg = p._get_initial_message(rng=np.random.default_rng(0))
    assert isinstance(msg, UncertainArray)
    assert np.allclose(msg.data, 9.0)
    assert msg.precision_mode == mode


@pytest.mark.parametrize("mode", [PrecisionMode.SCALAR, PrecisionMode.ARRAY])
def test_get_initial_message_sample_strategy(mode):
    """Sample strategy should use get_sample_for_output and match precision mode."""
    DummyPrior.called_get_sample = False
    p = DummyPrior(event_shape=(2, 2), precision_mode=mode)
    p.set_init_strategy("sample")
    msg = p._get_initial_message(np.random.default_rng(42))
    assert DummyPrior.called_get_sample is True
    assert isinstance(msg, UncertainArray)
    assert np.allclose(msg.data, 1.0)
    assert msg.precision_mode == mode


@pytest.mark.parametrize("mode", [PrecisionMode.SCALAR, PrecisionMode.ARRAY])
def test_get_initial_message_uninformative_strategy(mode, monkeypatch):
    """Uninformative strategy should call UncertainArray.random and match precision mode."""
    called = {}

    def mock_random(*args, **kwargs):
        called["ok"] = True
        ua = UncertainArray.zeros(kwargs["event_shape"], batch_size=kwargs["batch_size"])
        ua = ua.as_scalar_precision() if kwargs.get("scalar_precision", False) else ua.as_array_precision()
        return ua

    monkeypatch.setattr(UncertainArray, "random", mock_random)

    p = DummyPrior(event_shape=(3, 3), precision_mode=mode)
    p.set_init_strategy("uninformative")
    rng = np.random.default_rng(99)
    msg = p._get_initial_message(rng)
    assert called.get("ok", False)
    assert isinstance(msg, UncertainArray)
    assert msg.data.shape == (1, 3, 3)
    assert msg.precision_mode == mode


def test_get_initial_message_manual_without_message_raises():
    """If manual strategy selected but no message set, it should raise."""
    p = DummyPrior(event_shape=(2, 2), precision_mode=PrecisionMode.SCALAR)
    p.set_init_strategy("manual")
    with pytest.raises(RuntimeError, match="Manual initialization selected"):
        _ = p._get_initial_message(np.random.default_rng(0))


def test_get_initial_message_sample_not_implemented_raises():
    """If sample strategy selected but subclass does not implement sampler, raise RuntimeError."""
    class NoSamplePrior(Prior):
        def _compute_message(self, incoming): return incoming
        def get_sample_for_output(self, rng=None): raise NotImplementedError

    p = NoSamplePrior(event_shape=(2, 2), precision_mode=PrecisionMode.SCALAR)
    p.set_init_strategy("sample")
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="does not implement get_sample_for_output"):
        _ = p._get_initial_message(rng)


def test_get_initial_message_invalid_strategy_raises():
    """If init strategy string is invalid, ValueError should be raised."""
    p = DummyPrior(event_shape=(2, 2), precision_mode=PrecisionMode.SCALAR)
    p._init_strategy = "invalid"
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unknown init strategy"):
        _ = p._get_initial_message(rng)


# ============================================================
# 3. forward() integration with different strategies
# ============================================================

@pytest.mark.parametrize("mode", [PrecisionMode.SCALAR, PrecisionMode.ARRAY])
def test_forward_manual_strategy(mode):
    """Forward should use manual initialization message and preserve precision mode."""
    p = DummyPrior(event_shape=(2, 2), precision_mode=mode)
    rng = np.random.default_rng(1)
    p.set_init_rng(rng)
    p.set_manual_init(np.full((1, 2, 2), 5.0))
    p.set_init_strategy("manual")

    captured = {}
    def fake_receive_message(sender, msg):
        captured["msg"] = msg

    p.output.receive_message = fake_receive_message  # type: ignore
    p.forward()

    msg = captured["msg"]
    assert isinstance(msg, UncertainArray)
    assert np.allclose(msg.data, 5.0)
    assert msg.precision_mode == mode


@pytest.mark.parametrize("mode", [PrecisionMode.SCALAR, PrecisionMode.ARRAY])
def test_forward_sample_strategy(mode):
    """Forward should use sampling initialization if selected."""
    p = DummyPrior(event_shape=(2, 2), precision_mode=mode)
    p.set_init_strategy("sample")
    rng = np.random.default_rng(0)
    p.set_init_rng(rng)

    captured = {}
    def fake_receive_message(sender, msg):
        captured["msg"] = msg

    p.output.receive_message = fake_receive_message  # type: ignore
    p.forward()
    msg = captured["msg"]
    assert np.allclose(msg.data, 1.0)
    assert msg.precision_mode == mode


@pytest.mark.parametrize("mode", [PrecisionMode.SCALAR, PrecisionMode.ARRAY])
def test_forward_uninformative_strategy(mode, monkeypatch):
    """Forward should use uninformative initialization if selected."""
    p = DummyPrior(event_shape=(2, 2), precision_mode=mode)
    p.set_init_strategy("uninformative")
    rng = np.random.default_rng(0)
    p.set_init_rng(rng)

    called = {}
    def mock_random(*args, **kwargs):
        called["ok"] = True
        ua = UncertainArray.zeros(kwargs["event_shape"], batch_size=kwargs["batch_size"])
        ua = ua.as_scalar_precision() if kwargs.get("scalar_precision", False) else ua.as_array_precision()
        return ua

    monkeypatch.setattr(UncertainArray, "random", mock_random)

    captured = {}
    def fake_receive_message(sender, msg):
        captured["msg"] = msg

    p.output.receive_message = fake_receive_message  # type: ignore
    p.forward()
    assert called.get("ok", False)
    msg = captured["msg"]
    assert msg.precision_mode == mode


def test_forward_uses_compute_message_after_first_call():
    """Second forward() call should invoke _compute_message, not reinitialize."""
    p = DummyPrior(event_shape=(2, 2), precision_mode=PrecisionMode.SCALAR)
    rng = np.random.default_rng(0)
    p.set_init_rng(rng)
    p.output_message = UncertainArray.zeros(event_shape=(2, 2), batch_size=1)
    called = {}
    def fake_compute_message(incoming):
        called["ok"] = True
        return incoming
    p._compute_message = fake_compute_message  # type: ignore
    p.forward()
    assert called.get("ok", False)
