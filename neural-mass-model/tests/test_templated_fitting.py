import numpy as np
from neural_mass import fit_schizophrenia, fit_depression


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
