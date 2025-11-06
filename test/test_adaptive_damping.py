# tests/core/test_adaptive_damping.py
import pytest
from gpie.core.adaptive_damping import AdaptiveDamping, DampingScheduleConfig


def test_initial_state():
    """Check default initialization."""
    cfg = DampingScheduleConfig()
    ad = AdaptiveDamping(cfg)
    assert ad.beta == pytest.approx(cfg.beta_max)
    assert len(ad.hist) == 0
    assert 0 <= (1.0 - ad.beta) <= 1.0  # damping in [0,1]


def test_pass_condition_increases_beta():
    """When J decreases, beta should increase (damping decreases)."""
    cfg = DampingScheduleConfig(G_pass=1.1, G_fail=0.5, T_beta=3)
    ad = AdaptiveDamping(cfg)

    J_vals = [5.0, 4.0, 3.0, 2.5]
    betas = []
    for J in J_vals:
        damping, repeat = ad.step(J)
        betas.append(ad.beta)
        assert not repeat  # pass condition should hold
        assert 0 <= damping <= 1
    # beta should be non-decreasing
    assert all(betas[i + 1] >= betas[i] for i in range(len(betas) - 1))


def test_fail_condition_decreases_beta():
    """When J increases, beta should shrink (damping increases) and repeat=True."""
    cfg = DampingScheduleConfig(G_pass=1.1, G_fail=0.5, T_beta=2)
    ad = AdaptiveDamping(cfg)

    # Feed a good J first, then a larger J to trigger fail
    ad.step(1.0)
    beta_before = ad.beta
    damping, repeat = ad.step(10.0)
    assert repeat is True
    assert ad.beta < beta_before  # beta decreased
    assert 0 <= damping <= 1


def test_beta_bounds():
    """Ensure beta stays within [beta_min, beta_max] range."""
    cfg = DampingScheduleConfig(G_pass=2.0, G_fail=0.1,
                                beta_min=0.05, beta_max=1.0)
    ad = AdaptiveDamping(cfg)

    # many passes -> should saturate at beta_max
    for _ in range(10):
        ad.step(0.1)
    assert ad.beta <= cfg.beta_max + 1e-12

    # many fails -> should saturate at beta_min
    for _ in range(10):
        ad.step(10.0)
    assert ad.beta >= cfg.beta_min - 1e-12


def test_history_window_affects_pass_fail():
    """Ensure that T_beta defines the comparison window."""
    cfg = DampingScheduleConfig(G_pass=1.1, G_fail=0.5, T_beta=2)
    ad = AdaptiveDamping(cfg)

    # Insert a decreasing then increasing sequence
    ad.step(1.0)
    ad.step(0.9)
    # Now J=1.2 is worse than last two -> should fail
    damping, repeat = ad.step(1.2)
    assert repeat is True

    # Another smaller J=0.8 should pass
    damping, repeat = ad.step(0.8)
    assert repeat is False


def test_reset_restores_state():
    """Ensure reset() clears beta and history."""
    cfg = DampingScheduleConfig(G_pass=1.2, G_fail=0.5, T_beta=2)
    ad = AdaptiveDamping(cfg)
    ad.step(1.0)
    ad.step(0.5)
    assert len(ad.hist) > 0
    ad.beta = 0.3
    ad.reset()
    assert ad.beta == pytest.approx(cfg.beta_max)
    assert len(ad.hist) == 0


def test_repr_contains_key_fields():
    """Ensure repr() string contains useful information."""
    ad = AdaptiveDamping(DampingScheduleConfig())
    r = repr(ad)
    for kw in ["beta", "hist_len", "cfg"]:
        assert kw in r
