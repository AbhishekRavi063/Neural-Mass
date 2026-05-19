from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.event_detection import mask_segments
from src.kcomplex_features import event_iou, teager_energy


WINDOW_DETECTOR_FEATURES = (
    "peak_to_peak",
    "negative_peak",
    "positive_peak",
    "neg_then_pos",
    "slope_max",
    "slope_mean",
    "delta_power",
    "theta_power",
    "sigma_power",
    "delta_ratio",
    "slow_sigma_ratio",
    "teager_mean",
    "teager_max",
    "wavelet_slow_energy",
    "wavelet_fast_energy",
    "wavelet_ratio",
    "rms",
    "zero_crossings",
    "skewness",
    "kurtosis",
    "local_contrast",
)


def bandpass_signal(signal, sfreq, low=0.3, high=30.0):
    signal = np.asarray(signal, dtype=float)
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal - np.median(signal))


def sliding_windows(signal, sfreq, window_seconds=0.5, overlap_seconds=0.25):
    signal = np.asarray(signal, dtype=float)
    window_size = int(round(window_seconds * sfreq))
    step = int(round((window_seconds - overlap_seconds) * sfreq))
    step = max(1, step)
    if len(signal) < window_size:
        return []
    return [(start, start + window_size) for start in range(0, len(signal) - window_size + 1, step)]


def slow_wave_candidate_windows(signal, sfreq, window_seconds=0.9, min_distance=0.35, max_candidates=260):
    signal = np.asarray(signal, dtype=float)
    filtered = bandpass_signal(signal, sfreq, low=0.3, high=8.0)
    prominence = max(1e-8, np.std(filtered) * 0.45)
    distance = max(1, int(round(min_distance * sfreq)))
    negative_peaks, _ = find_peaks(-filtered, prominence=prominence, distance=distance)
    positive_peaks, _ = find_peaks(filtered, prominence=prominence, distance=distance)
    peaks = np.unique(np.concatenate([negative_peaks, positive_peaks]))
    if len(peaks) > max_candidates:
        amplitudes = np.abs(filtered[peaks])
        keep = np.argsort(amplitudes)[-max_candidates:]
        peaks = np.sort(peaks[keep])

    half = int(round(window_seconds * sfreq / 2))
    windows = []
    for peak in peaks:
        start = int(max(0, peak - half))
        end = int(min(len(signal), start + 2 * half))
        start = max(0, end - 2 * half)
        if end - start >= int(round(0.45 * sfreq)):
            windows.append((start, end))
    return windows


def _safe_band_power(segment, sfreq, low, high):
    if len(segment) < 8:
        return 0.0
    frequencies, power = welch(segment - np.mean(segment), fs=sfreq, nperseg=min(128, len(segment)))
    band = (frequencies >= low) & (frequencies <= high)
    return float(np.sum(power[band]))


def _haar_energies(segment):
    values = np.asarray(segment, dtype=float)
    energies = []
    current = values.copy()
    for _ in range(4):
        if len(current) < 4:
            energies.append(0.0)
            continue
        if len(current) % 2:
            current = current[:-1]
        approx = (current[0::2] + current[1::2]) / np.sqrt(2)
        detail = (current[0::2] - current[1::2]) / np.sqrt(2)
        energies.append(float(np.mean(detail**2)))
        current = approx
    return energies


def extract_window_features(signal, start, end, sfreq):
    signal = np.asarray(signal, dtype=float)
    segment = signal[start:end]
    context_radius = int(round(2.0 * sfreq))
    context = signal[max(0, start - context_radius):min(len(signal), end + context_radius)]
    if len(context) == 0:
        context = segment

    centered = segment - np.median(context)
    context_std = np.std(context) or 1.0
    derivative = np.diff(centered)
    minimum_idx = int(np.argmin(centered))
    maximum_idx = int(np.argmax(centered))
    delta_power = _safe_band_power(centered, sfreq, 0.5, 4.0)
    theta_power = _safe_band_power(centered, sfreq, 4.0, 8.0)
    sigma_power = _safe_band_power(centered, sfreq, 11.0, 16.0)
    broad_power = _safe_band_power(centered, sfreq, 0.5, 20.0)
    teo = teager_energy(centered)
    wavelet = _haar_energies(centered)
    wavelet_slow = wavelet[-1] + wavelet[-2]
    wavelet_fast = wavelet[0] + wavelet[1]

    return np.asarray([
        float(np.max(centered) - np.min(centered)),
        float(abs(np.min(centered))),
        float(abs(np.max(centered))),
        1.0 if minimum_idx < maximum_idx else 0.0,
        float(np.max(np.abs(derivative))) * sfreq if len(derivative) else 0.0,
        float(np.mean(np.abs(derivative))) * sfreq if len(derivative) else 0.0,
        delta_power,
        theta_power,
        sigma_power,
        delta_power / (broad_power + 1e-8),
        delta_power / (sigma_power + 1e-8),
        float(np.mean(np.abs(teo))) if len(teo) else 0.0,
        float(np.max(np.abs(teo))) if len(teo) else 0.0,
        wavelet_slow,
        wavelet_fast,
        wavelet_slow / (wavelet_fast + 1e-8),
        float(np.sqrt(np.mean(centered**2))),
        float(np.sum(np.diff(np.signbit(centered)) != 0)),
        float(skew(centered)) if len(centered) > 2 else 0.0,
        float(kurtosis(centered, fisher=False)) if len(centered) > 3 else 0.0,
        float((np.max(centered) - np.min(centered)) / context_std),
    ], dtype=float)


def label_windows(windows, expert_events, sfreq, iou_threshold=0.15):
    labels = []
    for start, end in windows:
        event = {"onset": start / sfreq, "end": end / sfreq}
        best = 0.0
        for expert in expert_events:
            best = max(best, event_iou(event, expert))
        labels.append(1 if best >= iou_threshold else 0)
    return np.asarray(labels, dtype=int)


def build_window_dataset(signal, sfreq, expert_events, mode="peaks"):
    filtered = bandpass_signal(signal, sfreq)
    if mode == "sliding":
        windows = sliding_windows(filtered, sfreq)
    elif mode == "peaks":
        windows = slow_wave_candidate_windows(filtered, sfreq)
    else:
        raise ValueError("mode must be 'peaks' or 'sliding'.")
    X = np.asarray([extract_window_features(filtered, start, end, sfreq) for start, end in windows])
    y = label_windows(windows, expert_events, sfreq)
    return filtered, windows, X, y


def train_balanced_window_classifier(X, y, random_state=42):
    classifier = RandomForestClassifier(
        n_estimators=220,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), classifier)
    model.fit(X, y)
    return model


def _is_biphasic(signal: NDArray, start: int, end: int, min_ratio: float = 0.35) -> bool:
    """Return True if the segment has K-complex morphology: negative peak before positive.

    min_ratio: negative peak must be at least this fraction of positive peak amplitude.
    """
    segment = signal[start:end]
    if len(segment) < 4:
        return False
    neg_idx = int(np.argmin(segment))
    pos_idx = int(np.argmax(segment))
    if neg_idx >= pos_idx:
        return False
    neg_amp = abs(float(np.min(segment)))
    pos_amp = abs(float(np.max(segment)))
    return neg_amp >= min_ratio * pos_amp


def _sigma_dominance(signal: NDArray, start: int, end: int, sfreq: float) -> float:
    """Fraction of delta+sigma power that lies in the sigma (spindle) band."""
    segment = signal[start:end]
    delta = _safe_band_power(segment, sfreq, 0.5, 4.0)
    sigma = _safe_band_power(segment, sfreq, 11.0, 16.0)
    return sigma / (delta + sigma + 1e-10)


def windows_to_events(
    windows,
    probabilities,
    sfreq,
    threshold=0.58,
    n_samples=None,
    signal: NDArray | None = None,
    spindle_rejection: bool = True,
    spindle_threshold: float = 0.38,
    morphology_filter: bool = False,
):
    """Convert per-window probabilities to merged event dicts.

    spindle_rejection: reject windows where sigma-band dominates (spindles).
    morphology_filter: reject events lacking neg-then-pos shape (off by default;
        CZ-A1 K-complexes can have inverted polarity depending on recording setup).
    """
    if n_samples is None:
        n_samples = max(end for _, end in windows) if windows else 0
    mask = np.zeros(n_samples, dtype=bool)
    for (start, end), probability in zip(windows, probabilities):
        if probability >= threshold:
            if signal is not None and spindle_rejection:
                if _sigma_dominance(signal, start, end, sfreq) > spindle_threshold:
                    continue
            mask[start:end] = True
    events = mask_to_postprocessed_events(mask, sfreq)
    if signal is not None and morphology_filter:
        events = [
            ev for ev in events
            if _is_biphasic(
                signal,
                int(round(ev["onset"] * sfreq)),
                int(round(ev["end"] * sfreq)),
            )
        ]
    return events


def mask_to_postprocessed_events(mask, sfreq, min_duration=0.35, max_duration=2.4, merge_gap=0.30, padding=0.10):
    mask = np.asarray(mask, dtype=bool).copy()
    if not mask.any():
        return []

    max_gap = int(round(merge_gap * sfreq))
    for start, end in mask_segments(~mask):
        if start == 0 or end == len(mask) - 1:
            continue
        if end - start + 1 <= max_gap:
            mask[start:end + 1] = True

    pad = int(round(padding * sfreq))
    events = []
    for start, end in mask_segments(mask):
        start = max(0, start - pad)
        end = min(len(mask) - 1, end + pad)
        duration = (end - start + 1) / sfreq
        if duration < min_duration or duration > max_duration:
            continue
        events.append({
            "onset": start / sfreq,
            "end": (end + 1) / sfreq,
            "duration": duration,
        })
    return events
