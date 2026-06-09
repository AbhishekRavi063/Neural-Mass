from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import optuna
from numpy.typing import NDArray
from scipy.signal import welch

from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters
from neural_mass.detection.event_detection import spindle_detection, K_complex_detection, mask_segments
from neural_mass.detection.kcomplex_features import generate_multitaper_kcomplex_candidates

optuna.logging.set_verbosity(optuna.logging.WARNING)


FIT_PARAMETER_RANGES = {
    # Core oscillator parameters
    "adaptation_strength":      (0.25, 0.85),
    "cortex_to_thalamus":       (0.10, 0.75),
    "thalamus_to_cortex":       (0.08, 0.65),
    "reticular_inhibition":     (0.25, 0.95),
    "background_drive":         (-0.05, 0.45),
    "cortical_damping":         (0.08, 0.35),
    "spindle_damping":          (0.30, 0.90),
    # Thalamic loop
    "relay_to_reticular":       (0.30, 1.20),
    "spindle_feedback_gain":    (3.0,  15.0),
    "cortical_excitation_scale":(8.0,  28.0),
    # EEG proxy mixing
    "eeg_spindle_weight":       (0.05, 0.40),
    "spindle_drive_offset":     (0.20, 0.70),
    # Newly exposed parameters for research-grade delta/sigma peak matching & 1/f modeling
    "cortical_frequency":       (0.40, 1.30),
    "spindle_frequency":        (11.0, 15.5),
    "adaptation_tau":           (1.0,  3.5),
    "eeg_relay_weight":         (0.01, 0.20),
    "pink_noise_std":           (0.0,  0.02),
}


FEATURE_NAMES = (
    "peak_to_peak",
    "negative_peak",
    "positive_peak",
    "slow_power",
    "spindle_power",
    "slow_to_spindle_ratio",
)


def band_power(signal: NDArray, sfreq: float, low: float, high: float) -> float:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 8:
        return 0.0
    # nperseg scaled to 2 seconds of data so frequency resolution ≈ 0.5 Hz regardless of sfreq.
    nperseg = min(int(2 * sfreq), len(signal))
    frequencies, power = welch(signal - np.mean(signal), fs=sfreq, nperseg=nperseg)
    band = (frequencies >= low) & (frequencies <= high)
    return float(np.sum(power[band]))


def extract_window_features(signal: NDArray, sfreq: float) -> dict[str, float]:
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.median(signal)
    slow_power = band_power(centered, sfreq, 0.3, 2.0)
    spindle_power = band_power(centered, sfreq, 11.0, 16.0)
    return {
        "peak_to_peak": float(np.percentile(centered, 95) - np.percentile(centered, 5)),
        "negative_peak": float(abs(np.percentile(centered, 5))),
        "positive_peak": float(abs(np.percentile(centered, 95))),
        "slow_power": slow_power,
        "spindle_power": spindle_power,
        "slow_to_spindle_ratio": slow_power / (spindle_power + 1e-8),
    }


def aggregate_feature_dicts(feature_dicts: list[dict[str, float]]) -> dict[str, float]:
    if not feature_dicts:
        raise ValueError("At least one feature dictionary is required.")
    return {
        name: float(np.median([features[name] for features in feature_dicts]))
        for name in FEATURE_NAMES
    }


def normalize_features(
    features: dict[str, float], reference: dict[str, float] | None = None
) -> NDArray[np.float64]:
    reference = reference or features
    values = []
    for name in FEATURE_NAMES:
        value = max(features[name], 1e-10)
        scale = max(abs(reference[name]), 1e-10)
        values.append(np.log1p(value / scale))
    return np.asarray(values, dtype=float)


def simulate_features(
    parameters: ThalamocorticalParameters, seconds: float, sfreq: int, seed: int
) -> dict[str, float]:
    model = ThalamocorticalModel(parameters, seed=seed)
    raw = model.simulate(seconds=seconds)
    stride = max(1, int(round((1 / parameters.dt) / sfreq)))
    eeg = raw["eeg"][::stride]
    if parameters.pink_noise_std > 0:
        from neural_mass.models.thalamocortical_model import _generate_pink_noise
        rng = np.random.default_rng(seed)
        eeg = eeg + _generate_pink_noise(len(eeg), parameters.pink_noise_std, rng)
    return extract_window_features(eeg, sfreq)


def simulate_eeg(
    parameters: ThalamocorticalParameters, seconds: float, sfreq: int, seed: int
) -> NDArray[np.float64]:
    model = ThalamocorticalModel(parameters, seed=seed)
    raw = model.simulate(seconds=seconds)
    stride = max(1, int(round((1 / parameters.dt) / sfreq)))
    eeg = raw["eeg"][::stride]
    if parameters.pink_noise_std > 0:
        from neural_mass.models.thalamocortical_model import _generate_pink_noise
        rng = np.random.default_rng(seed)
        eeg = eeg + _generate_pink_noise(len(eeg), parameters.pink_noise_std, rng)
    return eeg


def _params_from_trial(
    trial: optuna.Trial, base: ThalamocorticalParameters
) -> ThalamocorticalParameters:
    values = asdict(base)
    for name, (low, high) in FIT_PARAMETER_RANGES.items():
        values[name] = trial.suggest_float(name, low, high)
    values["noise_std"] = 0.0
    return ThalamocorticalParameters(**values)


def _best_params_from_study(
    study: optuna.Study, base: ThalamocorticalParameters, original_noise_std: float
) -> ThalamocorticalParameters:
    values = asdict(base)
    for name in FIT_PARAMETER_RANGES:
        values[name] = study.best_params[name]
    # Restore the original noise_std so that simulations with the fitted
    # parameters are stochastic. noise_std=0.0 is used only during fitting
    # itself (deterministic objective).
    values["noise_std"] = original_noise_std
    return ThalamocorticalParameters(**values)


def fit_thalamocortical_features(
    target_features: dict[str, float],
    seconds: float,
    sfreq: int = 200,
    n_trials: int = 60,
    seed: int = 13,
) -> tuple[ThalamocorticalParameters, dict[str, float], float]:
    """Fit compact thalamocortical parameters to window-level EEG features using Optuna TPE."""
    base = ThalamocorticalParameters(noise_std=0.0)
    target_vector = normalize_features(target_features, target_features)

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, base)
        candidate_features = simulate_features(params, seconds, sfreq, seed + trial.number)
        candidate_vector = normalize_features(candidate_features, target_features)
        return float(np.mean((candidate_vector - target_vector) ** 2))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    original_noise_std = ThalamocorticalParameters().noise_std
    best_parameters = _best_params_from_study(study, base, original_noise_std)
    best_features = simulate_features(best_parameters, seconds, sfreq, seed)
    return best_parameters, best_features, study.best_value


def standardize_waveform(signal: NDArray) -> NDArray[np.float64]:
    signal = np.asarray(signal, dtype=float)
    centered = signal - np.mean(signal)
    scale = np.std(centered)
    if scale <= 1e-12:
        scale = 1.0
    return centered / scale


def fit_thalamocortical_waveform(
    target_waveform: NDArray,
    sfreq: int = 200,
    n_trials: int = 40,
    seed: int = 101,
) -> tuple[ThalamocorticalParameters, NDArray[np.float64], float]:
    """Waveform-level fit using normalized shape error and Optuna TPE."""
    target = standardize_waveform(target_waveform)
    seconds = len(target) / sfreq
    base = ThalamocorticalParameters(noise_std=0.0)

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, base)
        waveform = standardize_waveform(
            simulate_eeg(params, seconds, sfreq, seed + trial.number)[: len(target)]
        )
        if len(waveform) != len(target):
            return float("inf")
        direct_error = float(np.mean((waveform - target) ** 2))
        flipped_error = float(np.mean((-waveform - target) ** 2))
        return min(direct_error, flipped_error)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    original_noise_std = ThalamocorticalParameters().noise_std
    best_parameters = _best_params_from_study(study, base, original_noise_std)
    best_waveform_raw = standardize_waveform(
        simulate_eeg(best_parameters, seconds, sfreq, seed)[: len(target)]
    )
    if len(best_waveform_raw) == len(target):
        direct_error = float(np.mean((best_waveform_raw - target) ** 2))
        flipped_error = float(np.mean((-best_waveform_raw - target) ** 2))
        if flipped_error < direct_error:
            best_waveform = -best_waveform_raw
            best_error = flipped_error
        else:
            best_waveform = best_waveform_raw
            best_error = direct_error
    else:
        best_waveform = best_waveform_raw
        best_error = study.best_value

    return best_parameters, best_waveform, best_error


def build_condition_summary(windows: list[NDArray], sfreq: float) -> dict[str, float]:
    return aggregate_feature_dicts(
        [extract_window_features(window, sfreq) for window in windows]
    )


def parameter_difference(
    a: ThalamocorticalParameters, b: ThalamocorticalParameters
) -> dict[str, float]:
    return {
        name: float(getattr(a, name) - getattr(b, name))
        for name in FIT_PARAMETER_RANGES
    }


def with_overrides(
    parameters: ThalamocorticalParameters, **overrides
) -> ThalamocorticalParameters:
    return replace(parameters, **overrides)


# --- Multi-Objective Profile Fitting ---

PROFILE_FEATURE_NAMES = (
    "spectral_delta",
    "spectral_theta",
    "spectral_alpha",
    "spectral_beta",
    "stats_mean",
    "stats_std",
    "stats_min",
    "stats_max",
    "stats_q1",
    "stats_q3",
    "grapho_kcomplex",
    "grapho_spindle",
)


def extract_profile_features(signal: NDArray, sfreq: float) -> dict[str, float]:
    """Extract Delta/Theta/Alpha/Beta power, stats, and spindle/K-complex counts."""
    signal = np.asarray(signal, dtype=float)
    
    # 1. Spectral Features
    delta = band_power(signal, sfreq, 0.5, 4.0)
    theta = band_power(signal, sfreq, 4.0, 8.0)
    alpha = band_power(signal, sfreq, 8.0, 12.0)
    beta = band_power(signal, sfreq, 12.0, 30.0)
    
    # 2. Statistical Features
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    q1 = float(np.percentile(signal, 25))
    q3 = float(np.percentile(signal, 75))
    
    # 3. Grapho-elements Features
    # Spindles: use the amplitude-adaptive rule-based detector
    try:
        spindle_mask = spindle_detection(signal, sampling_frequency=int(sfreq))
        num_spindles = len(mask_segments(spindle_mask))
    except Exception:
        num_spindles = 0.0

    # K-complex grapho feature: count sustained negative deflections (DOWN states).
    # Classic K-complex detectors require discrete transient events with specific
    # µV thresholds — they fail on synthetic AU-scale signals.  Instead we count
    # "slow-wave events" (periods below -1σ lasting ≥ 0.3s), which is:
    #   • Scale-agnostic (works on AU synthetic and µV real EEG)
    #   • Physiologically equivalent for the model (cortical DOWN states ≈ K-complexes)
    #   • Non-zero on the model output (used by the 3rd fitting objective)
    try:
        signal_std = float(np.std(signal)) or 1.0
        slow_mask = signal < -1.0 * signal_std
        min_samples = int(0.3 * sfreq)
        num_kcomplexes = float(sum(
            1 for s, e in mask_segments(slow_mask) if e - s + 1 >= min_samples
        ))
    except Exception:
        num_kcomplexes = 0.0
        
    return {
        "spectral_delta": delta,
        "spectral_theta": theta,
        "spectral_alpha": alpha,
        "spectral_beta": beta,
        "stats_mean": mean,
        "stats_std": std,
        "stats_min": min_val,
        "stats_max": max_val,
        "stats_q1": q1,
        "stats_q3": q3,
        "grapho_kcomplex": float(num_kcomplexes),
        "grapho_spindle": float(num_spindles),
    }


def fit_thalamocortical_multi_objective(
    target_signal: NDArray,
    sfreq: int = 200,
    n_trials: int = 40,
    seed: int = 42,
) -> tuple[ThalamocorticalParameters, dict[str, float], float]:
    """Fit compact thalamocortical parameters to a target signal using multi-objective Optuna.
    
    Objectives:
      1. Spectral Error (cosine distance)
      2. Statistical Error (normalized Euclidean distance)
      3. Grapho-element Error (Euclidean distance on counts)
    """
    target_signal = np.asarray(target_signal, dtype=float)
    seconds = len(target_signal) / sfreq
    target_features = extract_profile_features(target_signal, sfreq)
    base = ThalamocorticalParameters(noise_std=0.0)
    
    # Convert targets to numpy vectors for easier distance math
    target_spectral = np.array([target_features[k] for k in PROFILE_FEATURE_NAMES[:4]])
    target_stats = np.array([target_features[k] for k in PROFILE_FEATURE_NAMES[4:10]])
    target_grapho = np.array([target_features[k] for k in PROFILE_FEATURE_NAMES[10:]])
    
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = _params_from_trial(trial, base)
        try:
            # Simulate candidate signal
            sim_eeg = simulate_eeg(params, seconds, sfreq, seed + trial.number)
            sim_features = extract_profile_features(sim_eeg, sfreq)
        except Exception:
            return 1e6, 1e6, 1e6
            
        sim_spectral = np.array([sim_features[k] for k in PROFILE_FEATURE_NAMES[:4]])
        sim_stats = np.array([sim_features[k] for k in PROFILE_FEATURE_NAMES[4:10]])
        sim_grapho = np.array([sim_features[k] for k in PROFILE_FEATURE_NAMES[10:]])
        
        # 1. Spectral Error (Cosine distance or normalized L2)
        norm_target_spec = np.linalg.norm(target_spectral) or 1.0
        norm_sim_spec = np.linalg.norm(sim_spectral) or 1.0
        spectral_err = 1.0 - float(np.dot(target_spectral, sim_spectral) / (norm_target_spec * norm_sim_spec))
        if np.isnan(spectral_err):
            spectral_err = 1.0
            
        # 2. Statistical Error (Normalized Euclidean)
        denom_stats = np.abs(target_stats) + 1e-8
        stats_err = float(np.sqrt(np.mean(((sim_stats - target_stats) / denom_stats) ** 2)))
        
        # 3. Grapho-elements Error (Normalized absolute diff)
        denom_grapho = target_grapho + 1.0
        grapho_err = float(np.sqrt(np.mean(((sim_grapho - target_grapho) / denom_grapho) ** 2)))
        
        return spectral_err, stats_err, grapho_err

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    # Select the "best" trial using L1 compromise programming (lowest sum of normalized objectives)
    best_trial = None
    min_l1_norm = float("inf")
    
    # Gather all completed trials with valid values
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    
    if valid_trials:
        # Extract all objective values to calculate scaling bounds
        obj_vals = np.array([t.values for t in valid_trials])
        min_vals = obj_vals.min(axis=0)
        max_vals = obj_vals.max(axis=0)
        ranges = max_vals - min_vals
        # Avoid division by zero
        ranges[ranges <= 1e-8] = 1.0
        
        for t in valid_trials:
            # Normalize objectives to [0, 1] range to avoid magnitude bias
            norm_vals = (np.array(t.values) - min_vals) / ranges
            l1_norm = float(np.sum(norm_vals))
            if l1_norm < min_l1_norm:
                min_l1_norm = l1_norm
                best_trial = t
                
    if best_trial is None and valid_trials:
        best_trial = valid_trials[0]
        
    original_noise_std = ThalamocorticalParameters().noise_std
    if best_trial is not None:
        values = asdict(base)
        for name in FIT_PARAMETER_RANGES:
            if name in best_trial.params:
                values[name] = best_trial.params[name]
        values["noise_std"] = original_noise_std
        best_parameters = ThalamocorticalParameters(**values)
    else:
        best_parameters = base
        
    best_features = extract_profile_features(
        simulate_eeg(best_parameters, seconds, sfreq, seed), sfreq
    )
    
    # Return best parameters, best features, and the L1 norm error
    return best_parameters, best_features, min_l1_norm


def _params_from_trial_templated(
    trial: optuna.Trial, base: ThalamocorticalParameters, custom_ranges: dict[str, tuple[float, float]]
) -> ThalamocorticalParameters:
    values = asdict(base)
    for name in FIT_PARAMETER_RANGES:
        low, high = custom_ranges.get(name, FIT_PARAMETER_RANGES[name])
        values[name] = trial.suggest_float(name, low, high)
    values["noise_std"] = 0.0
    return ThalamocorticalParameters(**values)


def fit_thalamocortical_multi_objective_templated(
    target_signal: NDArray,
    custom_ranges: dict[str, tuple[float, float]],
    sfreq: int = 200,
    n_trials: int = 40,
    seed: int = 42,
) -> tuple[ThalamocorticalParameters, dict[str, float], float]:
    """Fit compact thalamocortical parameters with custom parameter ranges (disease templates)."""
    target_signal = np.asarray(target_signal, dtype=float)
    seconds = len(target_signal) / sfreq
    target_features = extract_profile_features(target_signal, sfreq)
    base = ThalamocorticalParameters(noise_std=0.0)
    
    target_spectral = np.array([target_features[k] for k in PROFILE_FEATURE_NAMES[:4]])
    target_stats = np.array([target_features[k] for k in PROFILE_FEATURE_NAMES[4:10]])
    target_grapho = np.array([target_features[k] for k in PROFILE_FEATURE_NAMES[10:]])
    
    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = _params_from_trial_templated(trial, base, custom_ranges)
        try:
            sim_eeg = simulate_eeg(params, seconds, sfreq, seed + trial.number)
            sim_features = extract_profile_features(sim_eeg, sfreq)
        except Exception:
            return 1e6, 1e6, 1e6
            
        sim_spectral = np.array([sim_features[k] for k in PROFILE_FEATURE_NAMES[:4]])
        sim_stats = np.array([sim_features[k] for k in PROFILE_FEATURE_NAMES[4:10]])
        sim_grapho = np.array([sim_features[k] for k in PROFILE_FEATURE_NAMES[10:]])
        
        norm_target_spec = np.linalg.norm(target_spectral) or 1.0
        norm_sim_spec = np.linalg.norm(sim_spectral) or 1.0
        spectral_err = 1.0 - float(np.dot(target_spectral, sim_spectral) / (norm_target_spec * norm_sim_spec))
        if np.isnan(spectral_err):
            spectral_err = 1.0
            
        denom_stats = np.abs(target_stats) + 1e-8
        stats_err = float(np.sqrt(np.mean(((sim_stats - target_stats) / denom_stats) ** 2)))
        
        denom_grapho = target_grapho + 1.0
        grapho_err = float(np.sqrt(np.mean(((sim_grapho - target_grapho) / denom_grapho) ** 2)))
        
        return spectral_err, stats_err, grapho_err

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    
    best_trial = None
    min_l1_norm = float("inf")
    valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    
    if valid_trials:
        obj_vals = np.array([t.values for t in valid_trials])
        min_vals = obj_vals.min(axis=0)
        max_vals = obj_vals.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges <= 1e-8] = 1.0
        
        for t in valid_trials:
            norm_vals = (np.array(t.values) - min_vals) / ranges
            l1_norm = float(np.sum(norm_vals))
            if l1_norm < min_l1_norm:
                min_l1_norm = l1_norm
                best_trial = t
                
    if best_trial is None and valid_trials:
        best_trial = valid_trials[0]
        
    original_noise_std = ThalamocorticalParameters().noise_std
    if best_trial is not None:
        values = asdict(base)
        for name in FIT_PARAMETER_RANGES:
            if name in best_trial.params:
                values[name] = best_trial.params[name]
        values["noise_std"] = original_noise_std
        best_parameters = ThalamocorticalParameters(**values)
    else:
        best_parameters = base
        
    best_features = extract_profile_features(
        simulate_eeg(best_parameters, seconds, sfreq, seed), sfreq
    )
    
    return best_parameters, best_features, min_l1_norm


def fit_schizophrenia(
    target_signal: NDArray,
    sfreq: int = 200,
    n_trials: int = 40,
    seed: int = 42,
) -> tuple[ThalamocorticalParameters, dict[str, float], float]:
    """Fit parameters using a Schizophrenia-specific template.
    
    Constrains:
      - reticular_inhibition: lower range (0.15 - 0.45) representing TRN deficit.
      - cortical_excitation_scale: lower range (8.0 - 15.0) representing synaptic pruning.
    """
    custom_ranges = {
        "reticular_inhibition": (0.15, 0.45),
        "cortical_excitation_scale": (8.0, 15.0),
    }
    return fit_thalamocortical_multi_objective_templated(
        target_signal, custom_ranges, sfreq=sfreq, n_trials=n_trials, seed=seed
    )


def fit_depression(
    target_signal: NDArray,
    sfreq: int = 200,
    n_trials: int = 40,
    seed: int = 42,
) -> tuple[ThalamocorticalParameters, dict[str, float], float]:
    """Fit parameters using a Depression-specific template.
    
    Constrains:
      - spindle_drive_offset: higher range (0.45 - 0.80) representing reduced spindle trigger.
      - adaptation_strength: lower range (0.10 - 0.50) representing altered plasticity.
    """
    custom_ranges = {
        "spindle_drive_offset": (0.45, 0.80),
        "adaptation_strength": (0.10, 0.50),
    }
    return fit_thalamocortical_multi_objective_templated(
        target_signal, custom_ranges, sfreq=sfreq, n_trials=n_trials, seed=seed
    )
