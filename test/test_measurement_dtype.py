import pytest
import numpy as onp  # for dtype references only

from gpie.core.backend import np, set_backend
from gpie.graph.measurement.base import Measurement
from gpie.core.types import get_real_dtype, get_complex_dtype
from gpie.core.uncertain_array import UncertainArray
from gpie.graph.wave import Wave


class SameDtypeMeasurement(Measurement):
    expected_input_dtype = np().complexfloating
    expected_observed_dtype = np().complexfloating

    def _infer_observed_dtype_from_input(self, input_dtype):
        return input_dtype

    def _compute_message(self, incoming):
        return incoming

    def _generate_sample(self, rng):
        pass

    def compute_belief(self):
        pass

    def compute_fitness(self):
        pass


class RealObservedMeasurement(Measurement):
    expected_input_dtype = np().complexfloating
    expected_observed_dtype = np().floating

    def _infer_observed_dtype_from_input(self, input_dtype):
        return get_real_dtype(input_dtype)

    def _compute_message(self, incoming):
        return incoming

    def _generate_sample(self, rng):
        pass

    def compute_belief(self):
        pass

    def compute_fitness(self):
        pass


class InvalidObservedMeasurement(Measurement):
    expected_input_dtype = np().floating
    expected_observed_dtype = np().floating

    def _infer_observed_dtype_from_input(self, input_dtype):
        return get_complex_dtype(input_dtype)  # Intentionally invalid

    def _compute_message(self, incoming):
        return incoming

    def _generate_sample(self, rng):
        pass

    def compute_belief(self):
        pass

    def compute_fitness(self):
        pass


def test_same_dtype_measurement():
    set_backend(onp)
    wave = Wave(event_shape=(2,), dtype=np().complex64)
    m = SameDtypeMeasurement()
    m << wave
    assert m.input_dtype == np().dtype(np().complex64)
    assert m.observed_dtype == np().dtype(np().complex64)


def test_real_observed_measurement():
    set_backend(onp)
    wave = Wave(event_shape=(4,), dtype=np().complex128)
    m = RealObservedMeasurement()
    m << wave
    assert m.input_dtype == np().dtype(np().complex128)
    assert m.observed_dtype == np().dtype(np().float64)


def test_invalid_observed_dtype():
    set_backend(onp)
    wave = Wave(event_shape=(1,), dtype=np().float64)
    m = InvalidObservedMeasurement()
    with pytest.raises(TypeError, match="expects observed dtype compatible"):
        m << wave


def test_invalid_input_dtype():
    set_backend(onp)
    class FloatRequiredMeasurement(Measurement):
        expected_input_dtype = np().floating
        expected_observed_dtype = np().floating

        def _infer_observed_dtype_from_input(self, input_dtype):
            return input_dtype

        def _compute_message(self, incoming):
            return incoming

        def _generate_sample(self, rng):
            pass
        
        def compute_belief(self):
            pass

        def compute_fitness(self):
            pass

    wave = Wave(event_shape=(3,), dtype=np().int32)
    m = FloatRequiredMeasurement()
    with pytest.raises(TypeError, match="expects input dtype compatible"):
        m << wave


def test_observed_dtype_casting():
    set_backend(onp)
    wave = Wave(event_shape=(2,), dtype=np().complex128)
    m = SameDtypeMeasurement()
    m << wave

    obs_data = np().array([[1+2j, 3+4j]], dtype=np().complex128)
    m.set_observed(obs_data, precision=1.0)

    assert m.observed.dtype == m.observed_dtype
    assert m.observed_dtype == np().dtype(np().complex128)


def test_set_observed_scalar_precision():
    set_backend(onp)
    wave = Wave(event_shape=(2,), dtype=np().complex64)
    m = RealObservedMeasurement()
    m << wave

    obs = np().array([[1.0, 2.0]], dtype=np().float32)
    m.set_observed(obs, precision=42.0)

    assert isinstance(m.observed, UncertainArray)
    assert np().allclose(m.observed.precision(), 42.0)
    assert m.observed.event_shape == (2,)
    assert m.observed.batch_shape == (1,)


def test_set_observed_array_precision_no_mask():
    set_backend(onp)
    wave = Wave(event_shape=(2,), dtype=np().complex64)
    m = RealObservedMeasurement()
    m << wave

    obs = np().array([[1.0, 2.0]], dtype=np().float32)
    prec = np().array([[10.0, 20.0]])
    m.set_observed(obs, precision=prec)

    assert np().allclose(m.observed.precision(), prec)
    assert m.observed.event_shape == (2,)
    assert m.observed.batch_shape == (1,)


def test_set_observed_with_mask_array_precision():
    set_backend(onp)
    wave = Wave(event_shape=(2,), dtype=np().complex64)
    m = RealObservedMeasurement()
    m << wave

    mask = np().array([[True, False]])
    obs = np().array([[1.0, 2.0]], dtype=np().float32)
    m.set_precision_mode_forward()
    m.set_observed(obs, precision=100.0, mask=mask)

    prec = m.observed.precision()
    assert prec.shape == (1, 2)
    assert prec[0, 0] == pytest.approx(100.0)
    assert prec[0, 1] == pytest.approx(0.0)


def test_set_observed_dtype_casting_respected():
    set_backend(onp)
    wave = Wave(event_shape=(2,), dtype=np().complex64)
    m = SameDtypeMeasurement()
    m << wave

    obs = np().array([[1+2j, 3+4j]], dtype=np().complex64)
    m.set_observed(obs, precision=1.0)

    assert m.observed.dtype == np().complex64
    assert isinstance(m.observed, UncertainArray)


def test_set_observed_batched_false():
    set_backend(onp)
    wave = Wave(event_shape=(2,), dtype=np().complex64)
    m = RealObservedMeasurement()
    m << wave

    obs = np().array([1.0, 2.0], dtype=np().float32)
    m.set_observed(obs, precision=1.0, batched=False)

    assert m.observed.batch_shape == (1,)
    assert m.observed.event_shape == (2,)
    assert isinstance(m.observed, UncertainArray)