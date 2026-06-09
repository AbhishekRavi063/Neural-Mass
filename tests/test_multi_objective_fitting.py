import numpy as np
import pytest

from neural_mass import (
    JensenRitModel,
    ThalamocorticalSleepModel,
    extract_profile_features,
    fit_thalamocortical_multi_objective,
)
from neural_mass.inference.thalamocortical_fitting import PROFILE_FEATURE_NAMES


def test_extract_profile_features_returns_expected_keys():
    sfreq = 200
    t = np.arange(1000) / sfreq
    # Generate a composite signal containing delta (1Hz) and spindle (13Hz) frequencies
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.sin(2 * np.pi * 13.0 * t)

    features = extract_profile_features(signal, sfreq)

    assert set(features.keys()) == set(PROFILE_FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features.values())
    
    # Verify spectral powers are positive
    assert features["spectral_delta"] > 0
    assert features["spectral_theta"] > 0
    
    # Verify statistical features match expected bounds
    assert features["stats_max"] >= features["stats_min"]
    assert features["stats_q3"] >= features["stats_q1"]


def test_fit_thalamocortical_multi_objective_returns_valid_results():
    sfreq = 200
    t = np.arange(1000) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.sin(2 * np.pi * 13.0 * t)

    # Run with 2 trials to keep the test extremely fast
    best_params, best_features, error = fit_thalamocortical_multi_objective(
        signal, sfreq=sfreq, n_trials=2, seed=42
    )

    assert best_params is not None
    assert best_params.cortical_frequency > 0
    assert set(best_features.keys()) == set(PROFILE_FEATURE_NAMES)
    assert np.isfinite(error)


def test_sklearn_thalamocortical_sleep_model_fit():
    sfreq = 200
    t = np.arange(1000) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.sin(2 * np.pi * 13.0 * t)

    # Initialize model
    model = ThalamocorticalSleepModel(seed=42)
    
    # Run fit with 2 trials for speed
    model.fit(signal, sfreq=sfreq, n_trials=2)

    # Check that best_parameters_ property exposes the fitted dictionary
    best_dict = model.best_parameters_
    assert isinstance(best_dict, dict)
    assert "cortical_frequency" in best_dict
    assert "spindle_frequency" in best_dict
    
    # Check that the best_parameters alias works too
    assert model.best_parameters == best_dict


def test_jensen_rit_model_alias_exact_match():
    # Verify that JensenRitModel is indeed our ThalamocorticalSleepModel wrapper
    assert JensenRitModel is ThalamocorticalSleepModel
