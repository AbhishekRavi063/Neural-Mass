import numpy as np
import pytest
from neural_mass import (
    compute_so_pac,
    estimate_thalamic_gating,
    clinical_artifact_filter,
    fit_schizophrenia,
    fit_depression,
)


def test_compute_so_pac_coupled_signal():
    """Verify that compute_so_pac successfully detects coupling and preferred phase."""
    sfreq = 200.0
    t = np.arange(10 * sfreq) / sfreq
    
    # 1. 1 Hz Slow Oscillation (peak at t=0, 1, 2, ...)
    so = np.cos(2.0 * np.pi * 1.0 * t)
    
    # 2. 13 Hz Spindle amplitude envelope: peaks at SO peaks
    envelope = 0.5 * (1.0 + np.cos(2.0 * np.pi * 1.0 * t))
    spindle = envelope * np.sin(2.0 * np.pi * 13.0 * t)
    
    combined = so + spindle
    
    pac_results = compute_so_pac(combined, sfreq=sfreq, n_bins=18)
    
    assert pac_results["modulation_index"] > 0.05
    # The peak of the cosine SO is at phase 0.0 radians (0 degrees)
    # The preferred phase should be close to 0 degrees (within 30 degrees)
    assert abs(pac_results["preferred_phase_deg"]) <= 30.0
    assert len(pac_results["bin_amplitudes"]) == 18


def test_estimate_thalamic_gating_fraction():
    """Verify that estimate_thalamic_gating calculates correct fraction above threshold."""
    dummy_outputs = {
        "thalamic_reticular": np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.4, 0.65, 0.66, 0.5])
    }
    # Elements > 0.65: 0.7, 0.8, 0.9, 0.66 (4 out of 9)
    tgi = estimate_thalamic_gating(dummy_outputs, reticular_threshold=0.65)
    assert abs(tgi - 4.0 / 9.0) < 1e-5


def test_clinical_artifact_filter_interpolation():
    """Verify that clinical_artifact_filter identifies and interpolates high-variance artifact epochs."""
    sfreq = 100.0
    t = np.arange(10 * sfreq) / sfreq
    
    # Clean sine wave
    signal = np.sin(2.0 * np.pi * 2.0 * t)
    
    # Inject high-variance noise in Epoch 2 (seconds 4.0 to 6.0)
    rng = np.random.default_rng(42)
    signal[int(4.0 * sfreq) : int(6.0 * sfreq)] += rng.normal(0.0, 30.0, size=200)
    
    filtered = clinical_artifact_filter(signal, sfreq=sfreq, reject_std_threshold=1.8)
    
    # Verify that the high-variance noise in Epoch 2 was successfully rejected and interpolated.
    # The max amplitude should be low (close to the sine wave's amplitude), and all values finite.
    assert np.max(np.abs(filtered)) < 5.0
    assert np.isfinite(filtered).all()


def test_templated_fitting_bounds():
    """Verify that fit_schizophrenia and fit_depression run and yield parameters in custom bounds."""
    sfreq = 200
    t = np.arange(1 * sfreq) / sfreq
    target_signal = np.sin(2 * np.pi * 2 * t) + 0.1 * np.sin(2 * np.pi * 12 * t)
    
    # Run a quick 2-trial fit to verify the templates execute and respect the constraints
    fitted_params_sz, _, _ = fit_schizophrenia(target_signal, sfreq=sfreq, n_trials=2, seed=42)
    # Schizophrenia constraints: reticular_inhibition [0.15, 0.45], cortical_excitation_scale [8.0, 15.0]
    assert 0.15 <= fitted_params_sz.reticular_inhibition <= 0.45
    assert 8.0 <= fitted_params_sz.cortical_excitation_scale <= 15.0

    fitted_params_dp, _, _ = fit_depression(target_signal, sfreq=sfreq, n_trials=2, seed=42)
    # Depression constraints: spindle_drive_offset [0.45, 0.80], adaptation_strength [0.10, 0.50]
    assert 0.45 <= fitted_params_dp.spindle_drive_offset <= 0.80
    assert 0.10 <= fitted_params_dp.adaptation_strength <= 0.50
