import numpy as np
import pytest

from src.metrics import (
    calculate_correlation,
    calculate_rhythmicity,
    calculate_rmse,
    calculate_snr,
)


def test_rmse_and_correlation_for_matching_signals():
    signal = np.array([0.0, 1.0, 2.0, 3.0])

    assert calculate_rmse(signal, signal) == 0.0
    assert calculate_correlation(signal, signal) == pytest.approx(1.0)


def test_rmse_rejects_different_lengths():
    with pytest.raises(ValueError):
        calculate_rmse(np.array([1.0, 2.0]), np.array([1.0]))


def test_rhythmicity_prefers_periodic_signal_over_noise():
    rng = np.random.default_rng(42)
    times = np.linspace(0, 4, 400)
    periodic = np.sin(2 * np.pi * 6 * times)
    noise = rng.normal(0, 1, len(times))

    assert calculate_rhythmicity(periodic) > calculate_rhythmicity(noise)


def test_spectral_snr_prefers_clean_tone_over_noisy_tone():
    rng = np.random.default_rng(42)
    sampling_frequency = 100
    times = np.arange(1000) / sampling_frequency
    clean = np.sin(2 * np.pi * 10 * times)
    noisy = clean + rng.normal(0, 2, len(times))

    assert calculate_snr(clean, sampling_frequency) > calculate_snr(noisy, sampling_frequency)
