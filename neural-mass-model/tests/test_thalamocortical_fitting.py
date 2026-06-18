import numpy as np

from neural_mass.inference.thalamocortical_fitting import (
    FEATURE_NAMES,
    aggregate_feature_dicts,
    build_condition_summary,
    extract_window_features,
    fit_thalamocortical_features,
    fit_thalamocortical_waveform,
    parameter_difference,
    standardize_waveform,
)


def test_extract_window_features_returns_expected_keys():
    sfreq = 200
    t = np.arange(1000) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.sin(2 * np.pi * 13.0 * t)

    features = extract_window_features(signal, sfreq)

    assert set(features) == set(FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features.values())
    assert features["slow_power"] > 0
    assert features["spindle_power"] > 0


def test_aggregate_feature_dicts_uses_median():
    first = {name: 1.0 for name in FEATURE_NAMES}
    second = {name: 3.0 for name in FEATURE_NAMES}
    third = {name: 100.0 for name in FEATURE_NAMES}

    summary = aggregate_feature_dicts([first, second, third])

    assert all(value == 3.0 for value in summary.values())


def test_build_condition_summary_from_windows():
    sfreq = 200
    windows = [np.zeros(1000), np.ones(1000)]

    summary = build_condition_summary(windows, sfreq)

    assert set(summary) == set(FEATURE_NAMES)
    assert all(np.isfinite(value) for value in summary.values())


def test_feature_fitting_returns_parameters_and_error():
    sfreq = 200
    t = np.arange(1000) / sfreq
    signal = np.sin(2 * np.pi * 0.8 * t) + 0.1 * np.sin(2 * np.pi * 13.0 * t)
    target = extract_window_features(signal, sfreq)

    params, features, error = fit_thalamocortical_features(target, seconds=5.0, sfreq=sfreq, n_trials=2, seed=1)

    assert params.dt > 0
    assert set(features) == set(FEATURE_NAMES)
    assert np.isfinite(error)


def test_parameter_difference_contains_fit_parameters():
    sfreq = 200
    t = np.arange(1000) / sfreq
    target = extract_window_features(np.sin(2 * np.pi * 0.8 * t), sfreq)
    a, _, _ = fit_thalamocortical_features(target, seconds=5.0, sfreq=sfreq, n_trials=1, seed=1)
    b, _, _ = fit_thalamocortical_features(target, seconds=5.0, sfreq=sfreq, n_trials=1, seed=2)

    diff = parameter_difference(a, b)

    assert "cortex_to_thalamus" in diff
    assert "thalamus_to_cortex" in diff


def test_standardize_waveform_handles_constant_signal():
    standardized = standardize_waveform(np.ones(20))

    assert np.isfinite(standardized).all()
    assert np.allclose(standardized, 0.0)


def test_waveform_fitting_returns_waveform_and_error():
    sfreq = 100
    t = np.arange(300) / sfreq
    target = np.sin(2 * np.pi * 0.8 * t)

    params, waveform, error = fit_thalamocortical_waveform(target, sfreq=sfreq, n_trials=2, seed=3)

    assert params.dt > 0
    assert waveform.shape == target.shape
    assert np.isfinite(error)
