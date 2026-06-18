import numpy as np
import pytest

from kcomplex_detector.event_detection import K_complex_detection, spindle_detection, splindle_detection
from kcomplex_detector.utils.data_loader import generate_realistic_eeg


def test_spindle_detection_finds_sustained_sigma_burst():
    sampling_frequency = 100
    times = np.arange(5 * sampling_frequency) / sampling_frequency
    signal = 0.1 * np.sin(2 * np.pi * 2 * times)
    burst_mask = (times >= 2.0) & (times < 3.0)
    signal[burst_mask] += 5.0 * np.sin(2 * np.pi * 12 * times[burst_mask])

    detected = spindle_detection(signal, sampling_frequency=sampling_frequency)

    assert detected.shape == signal.shape
    assert detected[burst_mask].any()
    assert np.array_equal(detected, splindle_detection(signal, sampling_frequency))


def test_k_complex_detection_finds_large_slow_wave():
    sampling_frequency = 100
    signal, events = generate_realistic_eeg(
        sfreq=sampling_frequency,
        seed=42,
        return_events=True,
    )
    wave_mask = events["k_complex"]

    detected = K_complex_detection(signal, sampling_frequency=sampling_frequency)

    assert detected.shape == signal.shape
    assert detected[wave_mask].any()


def test_event_detectors_reject_non_1d_input():
    bad_signal = np.zeros((2, 10))

    with pytest.raises(ValueError):
        spindle_detection(bad_signal)

    with pytest.raises(ValueError):
        K_complex_detection(bad_signal)
