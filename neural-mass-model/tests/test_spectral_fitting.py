import numpy as np
from neural_mass import fit_thalamocortical_spectral
from neural_mass.inference.thalamocortical_fitting import simulate_eeg, band_power
from neural_mass.models.thalamocortical_model import ThalamocorticalParameters


def test_spectral_fit_runs_and_returns_profile():
    sfreq = 200
    t = np.arange(6 * sfreq) / sfreq
    # delta-dominant target with a small spindle, like NREM sleep
    target = np.sin(2 * np.pi * 1.0 * t) + 0.15 * np.sin(2 * np.pi * 13.0 * t)

    params, profile, error = fit_thalamocortical_spectral(
        target, sfreq=sfreq, n_trials=3, seed=7, sim_seconds=20.0
    )

    assert params.pink_noise_std > 0
    assert set(profile.keys()) == {"delta", "theta", "alpha", "beta"}
    assert abs(sum(profile.values()) - 1.0) < 1e-6   # profile is normalized
    assert np.isfinite(error)


def test_observation_lowpass_attenuates_high_beta():
    """The observation low-pass should roll off >20 Hz power while leaving the
    delta band essentially intact."""
    sfreq = 200
    base = ThalamocorticalParameters(pink_noise_std=0.05, eeg_lowpass_hz=0.0)
    filt = ThalamocorticalParameters(pink_noise_std=0.05, eeg_lowpass_hz=16.0)

    raw = simulate_eeg(base, 30.0, sfreq, seed=7)
    low = simulate_eeg(filt, 30.0, sfreq, seed=7)

    # High-beta power is strongly attenuated...
    assert band_power(low, sfreq, 20.0, 30.0) < 0.2 * band_power(raw, sfreq, 20.0, 30.0)
    # ...while the delta band is largely preserved.
    assert band_power(low, sfreq, 0.5, 4.0) > 0.7 * band_power(raw, sfreq, 0.5, 4.0)


def test_measurement_floor_lifts_high_frequency_floor():
    """The white measurement-noise floor should restore the flat high-frequency
    floor that the observation low-pass removes, without touching the delta band."""
    sfreq = 200
    no_floor = ThalamocorticalParameters(eeg_lowpass_hz=16.0, measurement_noise_std=0.0)
    floor = ThalamocorticalParameters(eeg_lowpass_hz=16.0, measurement_noise_std=0.02)

    a = simulate_eeg(no_floor, 30.0, sfreq, seed=7)
    b = simulate_eeg(floor, 30.0, sfreq, seed=7)

    # The floor lifts >20 Hz power back up...
    assert band_power(b, sfreq, 20.0, 30.0) > 5.0 * band_power(a, sfreq, 20.0, 30.0)
    # ...while the dominant delta band is essentially unchanged.
    assert abs(band_power(b, sfreq, 0.5, 4.0) - band_power(a, sfreq, 0.5, 4.0)) < 0.1 * band_power(a, sfreq, 0.5, 4.0)
