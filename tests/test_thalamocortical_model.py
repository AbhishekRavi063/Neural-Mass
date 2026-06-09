import numpy as np
from scipy.signal import welch

from neural_mass.models.thalamocortical_model import (
    ThalamocorticalModel,
    ThalamocorticalParameters,
    simulate_thalamocortical_sleep,
)


def _band_power(signal, sfreq, low, high):
    frequencies, power = welch(signal - np.mean(signal), fs=sfreq, nperseg=min(512, len(signal)))
    band = (frequencies >= low) & (frequencies <= high)
    return float(np.sum(power[band]))


def test_thalamocortical_simulation_returns_expected_signals():
    signals = simulate_thalamocortical_sleep(seconds=2, sampling_frequency=200, seed=1)

    assert set(signals) == {
        "eeg",
        "cortical_pyramidal",
        "cortical_interneuron",
        "thalamic_relay",
        "thalamic_reticular",
        "adaptation",
        "spindle",
    }
    assert len(signals["eeg"]) == 400
    assert np.isfinite(signals["eeg"]).all()


def test_thalamocortical_simulation_is_reproducible():
    first = simulate_thalamocortical_sleep(seconds=2, sampling_frequency=200, seed=2)
    second = simulate_thalamocortical_sleep(seconds=2, sampling_frequency=200, seed=2)

    np.testing.assert_allclose(first["eeg"], second["eeg"])


def test_thalamocortical_model_rejects_bad_stimulus_length():
    model = ThalamocorticalModel(ThalamocorticalParameters(dt=0.01), seed=1)

    try:
        model.simulate(seconds=1.0, stimuli=np.zeros(10))
    except ValueError as exc:
        assert "stimuli" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid stimulus length.")


def test_thalamocortical_output_has_sleep_relevant_bands():
    sfreq = 200
    signals = simulate_thalamocortical_sleep(seconds=10, sampling_frequency=sfreq, seed=3)
    eeg = signals["eeg"]

    slow_power = _band_power(eeg, sfreq, 0.3, 1.5)
    spindle_power = _band_power(eeg, sfreq, 11.0, 16.0)

    assert slow_power > 0
    assert spindle_power > 0
