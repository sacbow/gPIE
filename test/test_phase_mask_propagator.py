import numpy as np
import pytest

from gpie.core.uncertain_array import UncertainArray
from gpie.graph.wave import Wave
from gpie.graph.propagator.phase_mask_propagator import PhaseMaskPropagator
from gpie.core.backend import set_backend, np as backend_np
from gpie.core.rng_utils import get_rng
from gpie.core.linalg_utils import random_phase_mask


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("batch_size", [1, 4])
def test_phase_mask_forward_backward(dtype, batch_size):
    set_backend(np)
    rng = get_rng(seed=42)

    shape = (64, 64)
    B = batch_size
    H, W = shape

    # Step 1: Create random input UA
    ua = UncertainArray.random(
        event_shape=shape,
        batch_size=B,
        dtype=dtype,
        precision=1.0,
        scalar_precision=True,  # test scalar mode
        rng=rng
    )

    # Step 2: Create phase mask
    mask = random_phase_mask(shape=shape, dtype=dtype, rng=rng)
    prop = PhaseMaskPropagator(mask, dtype=dtype)

    # Step 3: Connect dummy Wave
    w = Wave(event_shape=shape, batch_size=B, dtype=dtype)
    _ = prop @ w  # triggers __matmul__ to check broadcast etc.

    # Step 4: Forward + Backward
    ua_y = prop._compute_forward({"input": ua})
    ua_rec = prop._compute_backward(ua_y)

    # Step 5: Compare
    x_true = ua.data
    x_rec = ua_rec.data

    abs_err = backend_np().abs(x_true - x_rec)
    rel_err = abs_err / (backend_np().abs(x_true) + 1e-8)

    mean_rel = backend_np().mean(rel_err)
    max_rel = backend_np().max(rel_err)

    print(f"[dtype={dtype.__name__}, batch={B}] rel_err: mean={mean_rel:.2e}, max={max_rel:.2e}")
    assert mean_rel < 1e-4
    assert max_rel < 5e-4
