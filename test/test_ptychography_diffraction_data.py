import pytest
import numpy as np
from gpie.core.backend import set_backend, np as backend_np
from gpie.imaging.ptychography.data.diffraction_data import DiffractionData


def test_diffraction_data_attributes():
    set_backend(np)
    arr = backend_np().ones((16, 16), dtype=backend_np().complex64)
    data = DiffractionData(position=(10, 20), diffraction=arr, meta={"id": 1})

    assert data.position == (10, 20)
    assert data.diffraction.shape == (16, 16)
    assert data.meta["id"] == 1


def test_intensity_computation():
    set_backend(np)
    arr = backend_np().ones((8, 8), dtype=backend_np().complex64) * (1 + 1j)
    data = DiffractionData(position=(0, 0), diffraction=arr)
    intensity = data.intensity()

    assert intensity.shape == (8, 8)
    assert np.allclose(intensity, 2.0)  # |1+1j|^2 = 2


def test_summary_string():
    set_backend(np)
    arr = backend_np().zeros((4, 4), dtype=backend_np().complex64)
    data = DiffractionData(position=(5, 5), diffraction=arr, meta={"label": "test"})
    summary_str = data.summary()
    assert "Pos=(5, 5)" in summary_str
    assert "shape=(4, 4)" in summary_str
    assert "test" in str(data.meta)


def test_show_runs_without_error():
    import matplotlib
    matplotlib.use("Agg")  # headless
    set_backend(np)

    arr = backend_np().ones((8, 8), dtype=backend_np().complex64)
    data = DiffractionData(position=(0, 0), diffraction=arr)
    ax = data.show()
    assert ax is not None
