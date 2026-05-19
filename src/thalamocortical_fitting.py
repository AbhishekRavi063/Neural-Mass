from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import optuna
from numpy.typing import NDArray
from scipy.signal import welch

from src.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters

optuna.logging.set_verbosity(optuna.logging.WARNING)


FIT_PARAMETER_RANGES = {
    "adaptation_strength": (0.25, 0.85),
    "cortex_to_thalamus": (0.10, 0.75),
    "thalamus_to_cortex": (0.08, 0.65),
    "reticular_inhibition": (0.25, 0.95),
    "background_drive": (-0.05, 0.45),
    "cortical_damping": (0.08, 0.35),
    "spindle_damping": (0.30, 0.90),
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
    frequencies, power = welch(
        signal - np.mean(signal), fs=sfreq, nperseg=min(512, len(signal))
    )
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
    return extract_window_features(eeg, sfreq)


def simulate_eeg(
    parameters: ThalamocorticalParameters, seconds: float, sfreq: int, seed: int
) -> NDArray[np.float64]:
    model = ThalamocorticalModel(parameters, seed=seed)
    raw = model.simulate(seconds=seconds)
    stride = max(1, int(round((1 / parameters.dt) / sfreq)))
    return raw["eeg"][::stride]


def _params_from_trial(
    trial: optuna.Trial, base: ThalamocorticalParameters
) -> ThalamocorticalParameters:
    values = asdict(base)
    for name, (low, high) in FIT_PARAMETER_RANGES.items():
        values[name] = trial.suggest_float(name, low, high)
    values["noise_std"] = 0.0
    return ThalamocorticalParameters(**values)


def _best_params_from_study(
    study: optuna.Study, base: ThalamocorticalParameters
) -> ThalamocorticalParameters:
    values = asdict(base)
    for name in FIT_PARAMETER_RANGES:
        values[name] = study.best_params[name]
    values["noise_std"] = 0.0
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

    best_parameters = _best_params_from_study(study, base)
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

    best_parameters = _best_params_from_study(study, base)
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
