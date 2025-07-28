def test_const_wave_to_backend_roundtrip():
    import numpy as np
    import cupy as cp
    from gpie.core import backend
    from gpie.graph.prior.const_wave import ConstWave

    # NumPyで初期化
    backend.set_backend(np)
    data = np.ones((2, 2), dtype=np.complex128)
    cw = ConstWave(data)
    
    # CuPyへ切替
    backend.set_backend(cp)
    cw.to_backend()
    assert isinstance(cw._data, cp.ndarray)
    assert cw.dtype == cp.complex128

    # NumPyへ戻す
    backend.set_backend(np)
    cw.to_backend()
    assert isinstance(cw._data, np.ndarray)
    assert cw.dtype == np.complex128
