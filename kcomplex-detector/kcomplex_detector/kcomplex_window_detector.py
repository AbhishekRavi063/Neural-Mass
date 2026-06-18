from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks, sosfiltfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kcomplex_detector.event_detection import mask_segments
from kcomplex_detector.utils.event_scoring import score_events
from kcomplex_detector.kcomplex_features import event_iou, teager_energy

# Canonical threshold grid used for CV-based selection.  Imported by benchmark
# scripts so there is a single source of truth.
_THRESHOLDS_GRID = [round(t, 2) for t in np.arange(0.25, 0.75, 0.05)]

# Idealized K-complex template
_T_IDEAL = np.linspace(0, 1, 100)
_IDEAL_KC = np.zeros(100)
_IDX_NEG = _T_IDEAL < 0.35
_IDEAL_KC[_IDX_NEG] = -np.sin(np.pi * _T_IDEAL[_IDX_NEG] / 0.35)
_IDX_POS = (_T_IDEAL >= 0.35) & (_T_IDEAL < 0.75)
_IDEAL_KC[_IDX_POS] = 0.6 * np.sin(np.pi * (_T_IDEAL[_IDX_POS] - 0.35) / 0.40)
_IDX_DECAY = _T_IDEAL >= 0.75
_IDEAL_KC[_IDX_DECAY] = 0.6 * np.cos(np.pi * (_T_IDEAL[_IDX_DECAY] - 0.75) / 0.50)
_IDEAL_KC = (_IDEAL_KC - np.mean(_IDEAL_KC)) / (np.std(_IDEAL_KC) + 1e-10)


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
    # Subject-normalised amplitude (z-scored relative to local context)
    "negative_peak_z",
    "positive_peak_z",
    "peak_asymmetry",
    # Spectral discriminability
    "slow_wave_ratio",
    # Temporal structure
    "autocorr_lag1",
    # Local context ratios (SOTA updates)
    "delta_power_ratio_to_context",
    "sigma_power_ratio_to_context",
    "variance_ratio_to_context",
    "entropy_ratio_to_context",
    "teager_ratio_to_context",
    # Hjorth parameters
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "line_length",
    "slope_asymmetry",
    "neg_phase_dur",
    "phase_ratio",
    "slope_down",
    "slope_up",
    "slope_decay",
    "slope_ratio_down_up",
    "slope_ratio_up_decay",
    "neg_peak_loc",
    "pos_peak_loc",
    "template_corr",
    "surrounding_alpha_delta_ratio",
    # Negative peak amplitude z-scored against ±30s rolling background std.
    # Captures whether the event stands out on a longer timescale than the
    # 5-second local_contrast window — helps small-but-real N2 K-complexes
    # that are undersized compared to the immediate 5s context.
    "neg_peak_amplitude_z_rolling",
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


def slow_wave_candidate_windows(signal, sfreq, window_seconds=0.9, min_distance=0.35, max_candidates=1000):
    """Generate candidate windows centred on prominent slow-wave peaks.

    Uses a lower prominence threshold (0.35 × std) to capture smaller K-complexes,
    with adaptive escalation to 0.50 × std when the signal is very noisy (>200 raw
    candidates), and deduplication of overlapping windows (>70% overlap kept only once).
    """
    signal = np.asarray(signal, dtype=float)
    filtered = bandpass_signal(signal, sfreq, low=0.3, high=8.0)
    distance = max(1, int(round(min_distance * sfreq)))
    signal_std = np.std(filtered)

    prominence = max(1e-8, signal_std * 0.35)
    negative_peaks, _ = find_peaks(-filtered, prominence=prominence, distance=distance)
    positive_peaks, _ = find_peaks(filtered, prominence=prominence, distance=distance)
    peaks = np.unique(np.concatenate([negative_peaks, positive_peaks]))

    # Adaptive: noisy signals produce too many candidates → raise threshold
    if len(peaks) > 200:
        prominence = max(1e-8, signal_std * 0.50)
        negative_peaks, _ = find_peaks(-filtered, prominence=prominence, distance=distance)
        positive_peaks, _ = find_peaks(filtered, prominence=prominence, distance=distance)
        peaks = np.unique(np.concatenate([negative_peaks, positive_peaks]))
    if len(peaks) > max_candidates:
        amplitudes = np.abs(filtered[peaks])
        keep = np.argsort(amplitudes)[-max_candidates:]
        peaks = np.sort(peaks[keep])

    half = int(round(window_seconds * sfreq / 2))
    min_len = int(round(0.45 * sfreq))
    raw_windows = []
    for peak in peaks:
        start = int(max(0, peak - half))
        end = int(min(len(signal), start + 2 * half))
        start = max(0, end - 2 * half)
        if end - start >= min_len:
            raw_windows.append((start, end))

    # Deduplicate: if two windows overlap by more than 70%, keep only the one
    # centred closer to the larger peak — avoids classifying the same event twice.
    if not raw_windows:
        return []
    windows = [raw_windows[0]]
    for start, end in raw_windows[1:]:
        prev_start, prev_end = windows[-1]
        overlap = max(0, min(end, prev_end) - max(start, prev_start))
        union = max(end, prev_end) - min(start, prev_start)
        if overlap / union > 0.70:
            # Keep the one with the larger absolute peak
            if np.max(np.abs(filtered[start:end])) > np.max(np.abs(filtered[prev_start:prev_end])):
                windows[-1] = (start, end)
        else:
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
    context_radius = int(round(5.0 * sfreq))
    context_start = max(0, start - context_radius)
    context_end = min(len(signal), end + context_radius)
    context = signal[context_start:context_end]
    if len(context) == 0:
        context = segment

    centered = segment - np.median(context)
    context_centered = context - np.median(context)
    segment_std = np.std(centered) or 1.0
    context_std = np.std(context_centered) or 1.0
    derivative = np.diff(centered)
    double_derivative = np.diff(derivative) if len(derivative) > 1 else np.zeros(1)
    minimum_idx = int(np.argmin(centered))
    maximum_idx = int(np.argmax(centered))
    delta_power = _safe_band_power(centered, sfreq, 0.5, 4.0)
    theta_power = _safe_band_power(centered, sfreq, 4.0, 8.0)
    sigma_power = _safe_band_power(centered, sfreq, 11.0, 16.0)
    broad_power = _safe_band_power(centered, sfreq, 0.5, 20.0)

    context_delta = _safe_band_power(context_centered, sfreq, 0.5, 4.0) or 1.0
    context_sigma = _safe_band_power(context_centered, sfreq, 11.0, 16.0) or 1.0
    
    teo = teager_energy(centered)
    context_teo = teager_energy(context_centered)
    teo_mean = float(np.mean(np.abs(teo))) if len(teo) else 0.0
    context_teo_mean = float(np.mean(np.abs(context_teo))) if len(context_teo) else 1.0

    wavelet = _haar_energies(centered)
    wavelet_slow = wavelet[-1] + wavelet[-2]
    wavelet_fast = wavelet[0] + wavelet[1]

    neg_amp = float(abs(np.min(centered)))
    pos_amp = float(abs(np.max(centered)))
    alpha_power = _safe_band_power(centered, sfreq, 8.0, 12.0)

    # Lag-1 autocorrelation: captures temporal smoothness vs spiky structure
    if len(centered) > 2:
        mu = np.mean(centered)
        c0 = np.mean((centered - mu) ** 2)
        autocorr_lag1 = float(np.mean((centered[:-1] - mu) * (centered[1:] - mu)) / (c0 + 1e-10))
    else:
        autocorr_lag1 = 0.0

    def calc_entropy(x):
        p = np.abs(x - np.mean(x))
        p = p / (np.sum(p) + 1e-10)
        return float(-np.sum(p * np.log(p + 1e-10)))

    seg_entropy = calc_entropy(centered)
    context_entropy = calc_entropy(context_centered) or 1.0

    # Hjorth Parameters
    hjorth_activity = float(np.var(centered))
    deriv_std = np.std(derivative) or 1e-10
    hjorth_mobility = float(deriv_std / segment_std)
    double_deriv_std = np.std(double_derivative) or 1e-10
    double_deriv_mobility = double_deriv_std / deriv_std
    hjorth_complexity = float(double_deriv_mobility / (hjorth_mobility + 1e-10))

    # Line Length
    line_length = float(np.sum(np.abs(derivative)))

    # Slope Asymmetry
    neg_slopes = derivative[derivative < 0]
    pos_slopes = derivative[derivative > 0]
    max_neg_slope = float(abs(np.min(neg_slopes))) if len(neg_slopes) else 0.0
    max_pos_slope = float(np.max(pos_slopes)) if len(pos_slopes) else 1e-10
    slope_asymmetry = max_neg_slope / max_pos_slope

    # Phase Durations
    neg_phase_dur = minimum_idx / sfreq
    pos_phase_dur = (len(centered) - minimum_idx) / sfreq
    phase_ratio = neg_phase_dur / (pos_phase_dur + 1e-10)

    # Detailed slope dynamics
    slope_down = neg_amp / (neg_phase_dur + 1e-8)
    dur_up = abs(maximum_idx - minimum_idx) / sfreq
    slope_up = (neg_amp + pos_amp) / (dur_up + 1e-8)
    dur_decay = (len(centered) - maximum_idx) / sfreq
    slope_decay = pos_amp / (dur_decay + 1e-8)
    slope_ratio_down_up = slope_down / (slope_up + 1e-8)
    slope_ratio_up_decay = slope_up / (slope_decay + 1e-8)

    # Peak locations
    neg_peak_loc = minimum_idx / len(centered)
    pos_peak_loc = maximum_idx / len(centered)

    # Template correlation
    if len(centered) >= 8:
        interp_seg = np.interp(np.linspace(0, len(centered)-1, 100), np.arange(len(centered)), centered)
        interp_seg = (interp_seg - np.mean(interp_seg)) / (np.std(interp_seg) + 1e-10)
        template_corr = float(np.corrcoef(interp_seg, _IDEAL_KC)[0, 1])
    else:
        template_corr = 0.0

    # Surrounding sleep-stage proxy: alpha/delta power ratio over ±20 seconds.
    # High values indicate Wakefulness or N1; low values indicate N2/N3.
    long_radius = int(round(20.0 * sfreq))
    long_ctx = signal[max(0, start - long_radius): min(len(signal), end + long_radius)]
    if len(long_ctx) >= int(sfreq * 4):
        lfreqs, lpsd = welch(long_ctx - np.median(long_ctx), fs=sfreq, nperseg=min(512, len(long_ctx)))
        surrounding_alpha_delta_ratio = (
            float(np.sum(lpsd[(lfreqs >= 8.0) & (lfreqs <= 13.0)]))
            / (float(np.sum(lpsd[(lfreqs >= 0.5) & (lfreqs <= 4.0)])) + 1e-10)
        )
    else:
        surrounding_alpha_delta_ratio = 0.0

    # Rolling background std over ±30s: z-score neg_amp against this longer
    # baseline so small K-complexes buried in a quiet N2 epoch still score high.
    rolling_radius = int(round(30.0 * sfreq))
    rolling_ctx = signal[max(0, start - rolling_radius): min(len(signal), end + rolling_radius)]
    rolling_std = float(np.std(rolling_ctx)) if len(rolling_ctx) >= int(sfreq * 2) else context_std
    rolling_std = rolling_std or 1.0
    neg_peak_amplitude_z_rolling = neg_amp / rolling_std

    return np.asarray([
        float(np.max(centered) - np.min(centered)),
        neg_amp,
        pos_amp,
        1.0 if minimum_idx < maximum_idx else 0.0,
        float(np.max(np.abs(derivative))) * sfreq if len(derivative) else 0.0,
        float(np.mean(np.abs(derivative))) * sfreq if len(derivative) else 0.0,
        delta_power,
        theta_power,
        sigma_power,
        delta_power / (broad_power + 1e-8),
        delta_power / (sigma_power + 1e-8),
        teo_mean,
        float(np.max(np.abs(teo))) if len(teo) else 0.0,
        wavelet_slow,
        wavelet_fast,
        wavelet_slow / (wavelet_fast + 1e-8),
        float(np.sqrt(np.mean(centered**2))),
        float(np.sum(np.diff(np.signbit(centered)) != 0)),
        float(skew(centered)) if len(centered) > 2 else 0.0,
        float(kurtosis(centered, fisher=False)) if len(centered) > 3 else 0.0,
        float((np.max(centered) - np.min(centered)) / context_std),
        # Subject-normalised (z-scored relative to local context std)
        neg_amp / context_std,
        pos_amp / context_std,
        neg_amp / (pos_amp + 1e-8),  # K-complexes are typically neg-dominant (>1.0)
        # Spectral: delta dominance over theta+alpha (K-complexes are slow-wave)
        delta_power / (theta_power + alpha_power + 1e-8),
        autocorr_lag1,
        
        # --- CONTEXT RATIO FEATURES ---
        delta_power / context_delta,
        sigma_power / context_sigma,
        (segment_std**2) / (context_std**2),
        seg_entropy / context_entropy,
        teo_mean / context_teo_mean,

        # --- HJORTH PARAMETERS ---
        hjorth_activity,
        hjorth_mobility,
        hjorth_complexity,
        line_length,
        slope_asymmetry,
        neg_phase_dur,
        phase_ratio,

        # --- SUPER FEATURES ---
        slope_down,
        slope_up,
        slope_decay,
        slope_ratio_down_up,
        slope_ratio_up_decay,
        neg_peak_loc,
        pos_peak_loc,
        template_corr,
        surrounding_alpha_delta_ratio,
        neg_peak_amplitude_z_rolling,
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
    classifier = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.04,
        max_depth=6,
        l2_regularization=1.5,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=random_state,
        early_stopping=False,
    )
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(X, y)
    return model


def select_threshold_by_cv(datasets, thresholds=None, random_state=100, beta=2.0):
    """Select the threshold that maximises mean F-beta across leave-one-out folds.

    Uses F-beta (default beta=2) which weights recall beta^2 times more than
    precision. This directly addresses the FN bottleneck identified in the DREAMS
    diagnostic: 80% of missed events pass all rule gates but score below the
    F1-optimal threshold.

    Parameters
    ----------
    datasets : list of dicts with keys X, y, windows, filtered, sfreq, expert_events, signal
    thresholds : sequence of float or None (defaults to grid in _THRESHOLDS_GRID)
    beta : float
        F-beta weight. beta=1 → F1 (precision==recall). beta=2 → recall
        weighted 4× more than precision. Default: 2.0.

    Returns
    -------
    best_threshold : float
    """
    if thresholds is None:
        thresholds = _THRESHOLDS_GRID

    n = len(datasets)
    threshold_scores = {t: [] for t in thresholds}

    for test_idx in range(n):
        train_sets = [d for i, d in enumerate(datasets) if i != test_idx]
        X_train = np.vstack([d["X"] for d in train_sets])
        y_train = np.concatenate([d["y"] for d in train_sets])
        model = train_balanced_window_classifier(X_train, y_train, random_state=random_state + test_idx)

        test = datasets[test_idx]
        probs = model.predict_proba(test["X"])[:, 1]
        sfreq = test["sfreq"]
        n_samples = len(test["signal"])
        filtered = test["filtered"]
        expert_events = test["expert_events"]

        for t in thresholds:
            detected = windows_to_events(
                test["windows"], probs, sfreq,
                threshold=t, n_samples=n_samples,
                signal=filtered, spindle_rejection=True,
            )
            score = score_events(expert_events, detected)
            tp = score["tp"]; fp = score["fp"]; fn = score["fn"]
            prec = tp / (tp + fp + 1e-10)
            rec  = tp / (tp + fn + 1e-10)
            fbeta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-10)
            threshold_scores[t].append(fbeta)

    mean_scores = {t: float(np.mean(vals)) for t, vals in threshold_scores.items()}
    best_threshold = max(mean_scores, key=mean_scores.get)
    return best_threshold, mean_scores


def _is_biphasic(signal: NDArray, start: int, end: int, min_ratio: float = 0.35) -> bool:
    """Return True if the segment has K-complex morphology: negative peak before positive."""
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


def _high_freq_artifact(signal: NDArray, start: int, end: int, sfreq: float, threshold: float = 0.55) -> bool:
    """Return True if the window is dominated by high-frequency (>30 Hz) energy — EMG/artifact."""
    segment = signal[start:end]
    if len(segment) < 8:
        return False
    hf_power = _safe_band_power(segment, sfreq, 30.0, min(80.0, sfreq / 2 - 1))
    total_power = _safe_band_power(segment, sfreq, 0.5, min(80.0, sfreq / 2 - 1))
    return (hf_power / (total_power + 1e-10)) > threshold


def windows_to_events(
    windows,
    probabilities,
    sfreq,
    threshold=0.70,
    n_samples=None,
    signal: NDArray | None = None,
    spindle_rejection: bool = True,
    spindle_threshold: float = 0.38,
    morphology_filter: bool = False,
    artifact_rejection: bool = True,
    artifact_threshold: float = 0.55,
):
    """Convert per-window probabilities to merged event dicts.

    spindle_rejection: reject windows where sigma-band dominates (spindles).
    artifact_rejection: reject windows dominated by high-frequency EMG/artifact energy.
    morphology_filter: reject events lacking neg-then-pos shape (off by default;
        CZ-A1 K-complexes can have inverted polarity depending on recording setup).
    """
    if n_samples is None:
        n_samples = max(end for _, end in windows) if windows else 0
    mask = np.zeros(n_samples, dtype=bool)
    for (start, end), probability in zip(windows, probabilities):
        if probability >= threshold:
            if signal is not None:
                if spindle_rejection and _sigma_dominance(signal, start, end, sfreq) > spindle_threshold:
                    continue
                if artifact_rejection and _high_freq_artifact(signal, start, end, sfreq, artifact_threshold):
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


def mask_to_postprocessed_events(mask, sfreq, min_duration=0.50, max_duration=2.4, merge_gap=0.30, padding=0.10):
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
