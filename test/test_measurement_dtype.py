import pytest
import numpy as onp  # host numpy
from gpie.graph.measurement.base import Measurement
from gpie.core.types import get_real_dtype, get_complex_dtype
from gpie.core.uncertain_array import UncertainArray
from gpie.graph.wave import Wave


class SameDtypeMeasurement(Measurement):
    expected_input_dtype = onp.complexfloating
    expected_observed_dtype = onp.complexfloating

    def _infer_observed_dtype_from_input(self, input_dtype):
        return input_dtype

    def _compute_message(self, incoming):
        return incoming

    def _generate_sample(self, rng):
        pass


class RealObservedMeasurement(Measurement):
    expected_input_dtype = onp.complexfloating
    expected_observed_dtype = onp.floating

    def _infer_observed_dtype_from_input(self, input_dtype):
        return get_real_dtype(input_dtype)

    def _compute_message(self, incoming):
        return incoming

    def _generate_sample(self, rng):
        pass


class InvalidObservedMeasurement(Measurement):
    expected_input_dtype = onp.floating
    expected_observed_dtype = onp.floating

    def _infer_observed_dtype_from_input(self, input_dtype):
        return get_complex_dtype(input_dtype)  

    def _compute_message(self, incoming):
        return incoming

    def _generate_sample(self, rng):
        pass


def test_same_dtype_measurement():
    wave = Wave(event_shape=(2,), dtype=onp.complex64)
    m = SameDtypeMeasurement()
    m << wave
    assert m.input_dtype == onp.dtype(onp.complex64)
    assert m.observed_dtype == onp.dtype(onp.complex64)



def test_real_observed_measurement():
    wave = Wave(event_shape=(4,), dtype=onp.complex128)
    m = RealObservedMeasurement()
    m << wave
    assert m.input_dtype == onp.dtype(onp.complex128)
    assert m.observed_dtype == onp.dtype(onp.float64)



def test_invalid_observed_dtype():
    wave = Wave(event_shape=(1,), dtype=onp.float64)
    m = InvalidObservedMeasurement()
    with pytest.raises(TypeError, match="expects observed dtype compatible"):
        m << wave


def test_invalid_input_dtype():
    class FloatRequiredMeasurement(Measurement):
        expected_input_dtype = onp.floating
        expected_observed_dtype = onp.floating

        def _infer_observed_dtype_from_input(self, input_dtype):
            return input_dtype

        def _compute_message(self, incoming):
            return incoming

        def _generate_sample(self, rng):
            pass

    wave = Wave(event_shape=(3,), dtype=onp.int32)
    m = FloatRequiredMeasurement()
    with pytest.raises(TypeError, match="expects input dtype compatible"):
        m << wave


def test_observed_dtype_casting():
    wave = Wave(event_shape=(2,), dtype=onp.complex128)
    m = SameDtypeMeasurement()
    m << wave

    obs_data = onp.array([[1+2j, 3+4j]], dtype=onp.complex128)
    m.set_observed(obs_data, precision=1.0)

    assert m.observed.dtype == m.observed_dtype
    assert m.observed_dtype == onp.dtype(onp.complex128)



def test_set_observed_scalar_precision():
    wave = Wave(event_shape=(2,), dtype=onp.complex64)
    m = RealObservedMeasurement()
    m @ wave

    obs = onp.array([[1.0, 2.0]], dtype=onp.float32)
    m.set_observed(obs, precision=42.0)

    assert isinstance(m.observed, UncertainArray)
    assert onp.allclose(m.observed.precision(), 42.0)
    assert m.observed.event_shape == (2,)
    assert m.observed.batch_shape == (1,)


def test_set_observed_array_precision_no_mask():
    wave = Wave(event_shape=(2,), dtype=onp.complex64)
    m = RealObservedMeasurement()
    m @ wave

    obs = onp.array([[1.0, 2.0]], dtype=onp.float32)
    prec = onp.array([[10.0, 20.0]])
    m.set_observed(obs, precision=prec)

    assert onp.allclose(m.observed.precision(), prec)
    assert m.observed.event_shape == (2,)
    assert m.observed.batch_shape == (1,)


def test_set_observed_with_mask_array_precision():
    wave = Wave(event_shape=(2,), dtype=onp.complex64)
    m = RealObservedMeasurement()
    m @ wave

    mask = onp.array([[True, False]])
    obs = onp.array([[1.0, 2.0]], dtype=onp.float32)
    m.set_precision_mode_forward()  # manually set precision_mode if needed
    m.set_observed(obs, precision=100.0, mask=mask)

    prec = m.observed.precision()
    assert prec.shape == (1, 2)
    assert prec[0, 0] == pytest.approx(100.0)
    assert prec[0, 1] == pytest.approx(0.0)



def test_set_observed_dtype_casting_respected():
    wave = Wave(event_shape=(2,), dtype=onp.complex64)
    m = SameDtypeMeasurement()
    m @ wave

    obs = onp.array([[1+2j, 3+4j]], dtype=onp.complex64)
    m.set_observed(obs, precision=1.0)

    assert m.observed.dtype == onp.complex64
    assert isinstance(m.observed, UncertainArray)


def test_set_observed_batched_false():
    wave = Wave(event_shape=(2,), dtype=onp.complex64)
    m = RealObservedMeasurement()
    m @ wave

    obs = onp.array([1.0, 2.0], dtype=onp.float32)  # shape = (2,)
    m.set_observed(obs, precision=1.0, batched=False)

    assert m.observed.batch_shape == (1,)
    assert m.observed.event_shape == (2,)
    assert isinstance(m.observed, UncertainArray)

