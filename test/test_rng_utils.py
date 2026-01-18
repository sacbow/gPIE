import importlib.util
import warnings
import pytest
import sys
import numpy as np

from gpie.core import backend
from gpie.core import rng_utils

# Check CuPy availability
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

import types

def test_get_rng_cupy_fallback(monkeypatch):
    """Test get_rng warns and falls back to numpy if CuPy not installed."""
    if has_cupy:
        pytest.skip("CuPy is available, no fallback to test")

    # CuPyをインポート不可にする
    monkeypatch.setitem(sys.modules, "cupy", None)

    # ダミーの CuPy backend (モジュールっぽく振る舞うオブジェクト)
    FakeCupy = types.SimpleNamespace(__name__="cupy", random=None)

    from gpie.core import backend as be
    be._backend = FakeCupy  # set_backendを通さず直接差し替える

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rng = rng_utils.get_rng(seed=1)
        # フォールバックとしてnumpy.Generatorが返ることを確認
        assert isinstance(rng, np.random.Generator)
        assert any("CuPy backend selected" in str(warn.message) for warn in w)





def test_get_rng_numpy():
    """Test get_rng returns NumPy RNG when backend is numpy."""
    backend.set_backend(np)
    rng = rng_utils.get_rng(seed=123)
    assert isinstance(rng, np.random.Generator)
    val = rng.integers(0, 10)
    assert 0 <= val < 10

@pytest.mark.skipif(not has_cupy, reason="CuPy required")
def test_get_rng_cupy_real():
    """Test get_rng returns CuPy RNG when CuPy backend is active."""
    backend.set_backend(cp)
    rng = rng_utils.get_rng(seed=1)
    assert isinstance(rng, cp.random.Generator)

def test_get_rng_invalid_backend():
    """Test get_rng raises for unsupported backend."""
    backend.set_backend(type("FakeBackend", (), {"__name__": "unknown"})())
    with pytest.raises(NotImplementedError):
        rng_utils.get_rng()


def test_ensure_rng_backend_warns_numpy():
    """Test _ensure_rng_backend warns if RNG mismatches NumPy backend."""
    backend.set_backend(np)
    rng = object()  # Not a numpy RNG
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = rng_utils._ensure_rng_backend(rng)
        assert isinstance(out, np.random.Generator)
        assert any("[rng_utils] RNG backend mismatch" in str(warn.message) for warn in w)


@pytest.mark.skipif(not has_cupy, reason="CuPy required")
def test_ensure_rng_backend_warns_cupy():
    """Test _ensure_rng_backend warns if RNG mismatches CuPy backend."""
    backend.set_backend(cp)
    rng = object()  # Not a cupy RNG
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = rng_utils._ensure_rng_backend(rng)
        assert isinstance(out, cp.random.Generator)
        assert any("[rng_utils] RNG backend mismatch" in str(warn.message) for warn in w)


def test_normal_choice_shuffle_uniform_numpy():
    """Test normal, choice, shuffle, and uniform functions under numpy backend."""
    backend.set_backend(np)
    rng = np.random.default_rng(0)

    vals = rng_utils.normal(rng, size=5)
    assert vals.shape == (5,)

    choices = rng_utils.choice(rng, [1, 2, 3], size=2)
    assert len(choices) == 2

    arr = np.arange(5)
    shuffled = rng_utils.shuffle(rng, arr.copy())
    assert set(shuffled) == set(arr)

    unif = rng_utils.uniform(rng, low=0.0, high=1.0, size=3)
    assert unif.shape == (3,)


@pytest.mark.skipif(not has_cupy, reason="CuPy required")
def test_normal_choice_shuffle_uniform_cupy():
    """Test normal, choice, shuffle, and uniform functions under cupy backend."""
    backend.set_backend(cp)
    rng = cp.random.default_rng(0)

    vals = rng_utils.normal(rng, size=5)
    assert vals.shape == (5,)

    choices = rng_utils.choice(rng, cp.array([1, 2, 3]), size=2)
    assert choices.shape == (2,)

    arr = cp.arange(5)
    shuffled = rng_utils.shuffle(rng, arr.copy())
    assert shuffled.shape == arr.shape

    unif = rng_utils.uniform(rng, low=0.0, high=1.0, size=3)
    assert unif.shape == (3,)
