import pytest
from gpie.core.backend import np, set_backend
from gpie.imaging.ptychography.data.diffraction_data import DiffractionData
from gpie.imaging.ptychography.data.dataset import PtychographyDataset


@pytest.fixture
def dummy_diffraction():
    return DiffractionData(position=(10, 20), diffraction=np().ones((16, 16), dtype=np().complex64))


def test_initialization():
    ds = PtychographyDataset()
    assert ds.obj is None
    assert ds.prb is None
    assert len(ds) == 0


def test_object_and_probe_shape():
    ds = PtychographyDataset()
    obj = np().zeros((32, 32), dtype=np().complex64)
    prb = np().zeros((16, 16), dtype=np().complex64)
    ds.set_object(obj)
    ds.set_probe(prb)
    assert ds.obj_shape == (32, 32)
    assert ds.prb_shape == (16, 16)


def test_add_single_and_multiple_diffraction(dummy_diffraction):
    ds = PtychographyDataset()
    ds.add_data(dummy_diffraction)
    assert len(ds) == 1
    ds.add_data([dummy_diffraction, dummy_diffraction])
    assert len(ds) == 3


def test_add_invalid_data_raises():
    ds = PtychographyDataset()
    with pytest.raises(TypeError):
        ds.add_data("invalid")


def test_clear_data(dummy_diffraction):
    ds = PtychographyDataset()
    ds.add_data([dummy_diffraction, dummy_diffraction])
    ds.clear_data()
    assert len(ds) == 0


def test_scan_positions_and_diffraction_patterns(dummy_diffraction):
    ds = PtychographyDataset()
    ds.add_data(dummy_diffraction)
    assert ds.scan_positions == [(10, 20)]
    assert ds.diffraction_patterns[0].shape == (16, 16)


def test_getitem_and_len(dummy_diffraction):
    ds = PtychographyDataset()
    ds.add_data(dummy_diffraction)
    assert ds[0] is dummy_diffraction
    assert len(ds) == 1


def test_sort_by_center_distance():
    ds = PtychographyDataset()
    ds.set_object(np().zeros((100, 100), dtype=np().complex64))
    ds.add_data([
        DiffractionData(position=(50, 50), diffraction=np().zeros((8, 8))),
        DiffractionData(position=(10, 10), diffraction=np().zeros((8, 8))),
    ])
    ds.sort_data("center_distance")
    assert ds.scan_positions[0] == (50, 50)  # closer to center


def test_sort_by_meta_key():
    ds = PtychographyDataset()
    d1 = DiffractionData(position=(0, 0), diffraction=np().zeros((8, 8)), meta={"sort_key": 10})
    d2 = DiffractionData(position=(0, 0), diffraction=np().zeros((8, 8)), meta={"sort_key": 5})
    ds.add_data([d1, d2])
    ds.sort_data("meta:sort_key")
    assert ds[0] is d2


def test_sort_by_callable():
    ds = PtychographyDataset()
    d1 = DiffractionData(position=(0, 0), diffraction=np().zeros((8, 8)))
    d2 = DiffractionData(position=(100, 100), diffraction=np().zeros((8, 8)))
    ds.add_data([d1, d2])
    ds.sort_data(key=lambda d: d.position[0] + d.position[1])
    assert ds[0] is d1


def test_sort_invalid_key_raises():
    ds = PtychographyDataset()
    ds.add_data(DiffractionData(position=(0, 0), diffraction=np().zeros((8, 8))))
    with pytest.raises(ValueError):
        ds.sort_data("unknown_key")
