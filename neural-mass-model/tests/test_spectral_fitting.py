import numpy as np
from neural_mass import fit_thalamocortical_spectral


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
