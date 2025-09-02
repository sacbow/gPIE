import importlib.util
import pytest
import numpy as np

from numpy.random import default_rng
import gpie
from gpie import Graph, SparsePrior, GaussianMeasurement, UnitaryPropagator, mse
from gpie.core.linalg_utils import random_unitary_matrix, random_binary_mask

# Optional CuPy support
cupy_spec = importlib.util.find_spec("cupy")
has_cupy = cupy_spec is not None
if has_cupy:
    import cupy as cp

backend_libs = [np]
if has_cupy:
    backend_libs.append(cp)

@pytest.mark.parametrize("xp", backend_libs)
def test_compressive_sensing_mse_decreases(xp):
    """Test compressive sensing graph inference under numpy/cupy backends."""
    gpie.set_backend(xp)

    # 1. パラメータ設定
    n = 512  # 短縮サイズでテスト高速化
    rho = 0.1
    var = 1e-4
    mask_ratio = 0.3

    # 2. ユニタリ行列 U とマスク生成
    rng = default_rng(seed=12)
    U = random_unitary_matrix(n, rng=rng, dtype=xp.complex128)
    mask = random_binary_mask(n, subsampling_rate=mask_ratio, rng=rng)

    # 3. グラフ構築
    class CompressiveSensingGraph(Graph):
        def __init__(self):
            super().__init__()
            x = ~SparsePrior(rho=rho, shape=(n,), damping=0.03, label="x", dtype=xp.complex128)
            with self.observe():
                GaussianMeasurement(var=var, mask=mask) @ (UnitaryPropagator(U) @ x)
            self.compile()

    g = CompressiveSensingGraph()

    # 4. RNG設定
    g.set_init_rng(xp.random.default_rng(seed=11))
    g.generate_sample(rng=xp.random.default_rng(seed=42))

    # 5. 真の信号取得
    true_x = g.get_wave("x").get_sample()

    # 6. 推論実行 + MSEモニタリング
    mse_list = []
    def monitor(graph, t):
        est = graph.get_wave("x").compute_belief().data
        err = mse(est, true_x)
        mse_list.append(err)

    g.run(n_iter=15, callback=monitor)

    # 7. MSEが単調減少（もしくは初期値より改善）することを確認
    assert mse_list[0] > mse_list[-1], "MSE should decrease during inference"

    # 8. 最終MSEが十分小さいことを確認
    final_est = g.get_wave("x").compute_belief().data
    final_mse = mse(final_est, true_x)
    assert final_mse < 1e-3, f"Final MSE too high: {final_mse}"
