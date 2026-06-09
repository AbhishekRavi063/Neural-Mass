import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal.windows import dpss

# np.trapz removed in NumPy 2.0; np.trapezoid introduced in NumPy 2.0.
try:
    _trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz       # NumPy <  2.0

from neural_mass.detection.event_detection import K_complex_detection


LOOSE_CANDIDATE_PARAMS = {
    "threshold_std": 1.35,
    "min_duration": 0.08,
    "merge_gap": 0.45,
    "event_padding": 0.20,
    "min_event_duration": 0.15,
    "max_event_duration": 2.5,
    "min_peak_to_peak": 0.0,
    "require_biphasic": False,
}


def mask_to_events(mask, sfreq):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    return [
        {
            "start": int(start),
            "end": int(end),
            "onset": start / sfreq,
            "end_time": (end + 1) / sfreq,
            "duration": (end - start + 1) / sfreq,
        }
        for start, end in zip(starts, ends)
    ]


def events_to_mask(events, n_samples, sfreq):
    mask = np.zeros(n_samples, dtype=bool)
    for event in events:
        start = int(round(event["onset"] * sfreq))
        end = int(round(event.get("end_time", event.get("end", event["onset"])) * sfreq))
        start = max(0, min(n_samples, start))
        end = max(start + 1, min(n_samples, end))
        mask[start:end] = True
    return mask


def _robust_zscore(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad if mad > 1e-12 else np.std(values)
    scale = scale if scale > 1e-12 else 1.0
    return (values - median) / scale


def _merge_overlapping_events(events, sfreq, max_gap=0.20):
    if not events:
        return []

    ordered = sorted(events, key=lambda event: event["onset"])
    merged = [ordered[0].copy()]
    for event in ordered[1:]:
        current = merged[-1]
        current_end = current["end_time"]
        if event["onset"] <= current_end + max_gap:
            current["end_time"] = max(current_end, event["end_time"])
            current["duration"] = current["end_time"] - current["onset"]
            current["start"] = int(round(current["onset"] * sfreq))
            current["end"] = int(round(current["end_time"] * sfreq)) - 1
        else:
            merged.append(event.copy())
    return merged


def _filter_event_shape(signal, events, sfreq, min_duration=0.35, max_duration=2.8, min_peak_to_peak=12.0):
    filtered = []
    signal = np.asarray(signal)
    for event in events:
        start = max(0, int(event["start"]))
        end = min(len(signal) - 1, int(event["end"]))
        if end <= start:
            continue

        duration = (end - start + 1) / sfreq
        if duration < min_duration or duration > max_duration:
            continue

        segment = signal[start:end + 1]
        peak_to_peak = float(np.max(segment) - np.min(segment))
        if peak_to_peak < min_peak_to_peak:
            continue

        minimum_idx = int(np.argmin(segment))
        maximum_idx = int(np.argmax(segment))
        has_kcomplex_order = minimum_idx < maximum_idx
        if not has_kcomplex_order and peak_to_peak < 2.0 * min_peak_to_peak:
            continue

        filtered.append({
            "start": start,
            "end": end,
            "onset": start / sfreq,
            "end_time": (end + 1) / sfreq,
            "duration": duration,
        })
    return filtered


def generate_multitaper_kcomplex_candidates(
    signal,
    sfreq,
    window_duration=1.10,
    step_duration=0.05,
    time_bandwidth=3.0,
    num_tapers=5,
    score_threshold=2.15,
    event_padding=0.25,
):
    """Generate loose K-complex candidates from multitaper delta-band bursts."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if len(signal) == 0:
        return []

    sfreq = float(sfreq)
    window_size = max(16, int(round(window_duration * sfreq)))
    step = max(1, int(round(step_duration * sfreq)))
    if len(signal) < window_size:
        return []

    sos = butter(4, [0.3, 18.0], btype="bandpass", output="sos", fs=sfreq)
    filtered = sosfiltfilt(sos, signal - np.median(signal))
    tapers = dpss(window_size, NW=time_bandwidth, Kmax=num_tapers, sym=False)
    frequencies = np.fft.rfftfreq(window_size, d=1 / sfreq)
    delta_band = (frequencies >= 0.5) & (frequencies <= 4.0)
    slow_band = (frequencies >= 0.5) & (frequencies <= 2.5)
    broad_band = (frequencies >= 0.5) & (frequencies <= 16.0)
    sigma_band = (frequencies >= 11.0) & (frequencies <= 16.0)

    starts = np.arange(0, len(filtered) - window_size + 1, step)
    delta_power = []
    slow_power = []
    delta_ratio = []
    sigma_ratio = []
    peak_to_peak = []
    neg_to_pos = []

    for start in starts:
        segment = filtered[start:start + window_size]
        segment = segment - np.mean(segment)
        spectra = np.fft.rfft(tapers * segment, axis=1)
        power = np.mean(np.abs(spectra) ** 2, axis=0)
        broad = float(np.sum(power[broad_band])) + 1e-10
        delta = float(np.sum(power[delta_band]))
        slow = float(np.sum(power[slow_band]))
        sigma = float(np.sum(power[sigma_band]))
        minimum_idx = int(np.argmin(segment))
        maximum_idx = int(np.argmax(segment))

        delta_power.append(delta)
        slow_power.append(slow)
        delta_ratio.append(delta / broad)
        sigma_ratio.append(sigma / broad)
        peak_to_peak.append(float(np.max(segment) - np.min(segment)))
        neg_to_pos.append(1.0 if minimum_idx < maximum_idx else 0.0)

    delta_power = np.asarray(delta_power)
    slow_power = np.asarray(slow_power)
    delta_ratio = np.asarray(delta_ratio)
    sigma_ratio = np.asarray(sigma_ratio)
    peak_to_peak = np.asarray(peak_to_peak)
    neg_to_pos = np.asarray(neg_to_pos)

    score = (
        0.85 * _robust_zscore(np.log1p(delta_power))
        + 0.55 * _robust_zscore(np.log1p(slow_power))
        + 0.75 * _robust_zscore(delta_ratio)
        + 0.55 * _robust_zscore(peak_to_peak)
        - 0.35 * _robust_zscore(sigma_ratio)
        + 0.25 * neg_to_pos
    )

    active_windows = score >= score_threshold
    peak_zscore = _robust_zscore(peak_to_peak)
    if not np.any(active_windows):
        active_windows = peak_zscore >= max(0.75, score_threshold)

    mask = np.zeros(len(signal), dtype=bool)
    padding = int(round(event_padding * sfreq))
    for start, active in zip(starts, active_windows):
        if active:
            left = max(0, start - padding)
            right = min(len(signal), start + window_size + padding)
            mask[left:right] = True

    events = _filter_event_shape(signal, mask_to_events(mask, sfreq), sfreq)
    if events or not np.any(active_windows):
        return events

    best_idx = int(np.argmax(score + 0.5 * peak_zscore))
    center = starts[best_idx] + window_size // 2
    half_width = int(round(0.75 * sfreq))
    fallback = [{
        "start": max(0, center - half_width),
        "end": min(len(signal) - 1, center + half_width),
    }]
    fallback[0]["onset"] = fallback[0]["start"] / sfreq
    fallback[0]["end_time"] = (fallback[0]["end"] + 1) / sfreq
    fallback[0]["duration"] = fallback[0]["end_time"] - fallback[0]["onset"]
    return _filter_event_shape(signal, fallback, sfreq, max_duration=3.2)


def generate_kcomplex_candidates(signal, sfreq, params=None, method="hybrid"):
    params = params or LOOSE_CANDIDATE_PARAMS
    if method not in {"rule", "multitaper", "hybrid"}:
        raise ValueError("method must be 'rule', 'multitaper', or 'hybrid'.")

    events = []
    if method in {"rule", "hybrid"}:
        mask = K_complex_detection(signal, sampling_frequency=int(sfreq), **params)
        events.extend(mask_to_events(mask, sfreq))

    if method in {"multitaper", "hybrid"}:
        events.extend(generate_multitaper_kcomplex_candidates(signal, sfreq))

    return _merge_overlapping_events(events, sfreq)


def _safe_power(segment, sfreq, low, high):
    if len(segment) < 8:
        return 0.0
    frequencies, power = welch(segment - np.mean(segment), fs=sfreq, nperseg=min(256, len(segment)))
    band = (frequencies >= low) & (frequencies <= high)
    return float(np.sum(power[band]))


def teager_energy(x):
    x = np.asarray(x)
    if len(x) < 3:
        return np.zeros(0)
    return x[1:-1] ** 2 - x[:-2] * x[2:]


def extract_kcomplex_features(signal, event, sfreq):
    signal = np.asarray(signal)
    start = max(0, event["start"])
    end = min(len(signal) - 1, event["end"])
    segment = signal[start:end + 1]
    if len(segment) == 0:
        return np.zeros(16)

    duration = (end - start + 1) / sfreq
    baseline_radius = int(2 * sfreq)
    context_start = max(0, start - baseline_radius)
    context_end = min(len(signal), end + baseline_radius + 1)
    context = signal[context_start:context_end]
    context_std = np.std(context) or 1.0

    minimum = float(np.min(segment))
    maximum = float(np.max(segment))
    peak_to_peak = maximum - minimum
    min_idx = int(np.argmin(segment))
    max_idx = int(np.argmax(segment))
    neg_then_pos = 1.0 if min_idx < max_idx else 0.0
    peak_gap = abs(max_idx - min_idx) / sfreq

    first = segment[: max(1, len(segment) // 2)]
    second = segment[max(1, len(segment) // 2):]
    first_min = float(np.min(first))
    second_max = float(np.max(second)) if len(second) else maximum

    derivative = np.diff(segment)
    max_abs_slope = float(np.max(np.abs(derivative))) * sfreq if len(derivative) else 0.0
    mean_abs_slope = float(np.mean(np.abs(derivative))) * sfreq if len(derivative) else 0.0
    area = float(_trapz(np.abs(segment - np.median(context)), dx=1 / sfreq))

    low_power = _safe_power(segment, sfreq, 0.5, 5.0)
    delta_power = _safe_power(segment, sfreq, 0.5, 4.0)
    theta_power = _safe_power(segment, sfreq, 4.0, 8.0)
    alpha_power = _safe_power(segment, sfreq, 8.0, 12.0)
    sigma_power = _safe_power(segment, sfreq, 11.0, 16.0)
    power_ratio = low_power / (sigma_power + 1e-8)

    zero_centered = segment - np.mean(context)
    zero_crossings = float(np.sum(np.diff(np.signbit(zero_centered)) != 0))
    teo = teager_energy(segment)
    context_teo = teager_energy(context)
    teo_mean = float(np.mean(np.abs(teo))) if len(teo) else 0.0
    teo_max = float(np.max(np.abs(teo))) if len(teo) else 0.0
    teo_ratio = teo_mean / (float(np.mean(np.abs(context_teo))) + 1e-8 if len(context_teo) else 1.0)

    rms = float(np.sqrt(np.mean(segment**2)))
    context_rms = float(np.sqrt(np.mean(context**2))) or 1.0
    percentile_span = float(np.percentile(segment, 95) - np.percentile(segment, 5))
    skewness = float(skew(segment)) if len(segment) > 2 else 0.0
    kurt = float(kurtosis(segment, fisher=False)) if len(segment) > 3 else 0.0
    entropy_power = np.abs(segment - np.mean(segment))
    entropy_power = entropy_power / (np.sum(entropy_power) + 1e-8)
    entropy = float(-np.sum(entropy_power * np.log(entropy_power + 1e-8)))

    return np.array([
        duration,
        peak_to_peak,
        abs(minimum),
        abs(maximum),
        abs(first_min),
        abs(second_max),
        peak_to_peak / context_std,
        abs(minimum) / context_std,
        abs(maximum) / context_std,
        neg_then_pos,
        peak_gap,
        max_abs_slope,
        mean_abs_slope,
        area,
        power_ratio,
        zero_crossings,
        delta_power,
        theta_power,
        alpha_power,
        sigma_power,
        low_power / (theta_power + alpha_power + sigma_power + 1e-8),
        teo_mean,
        teo_max,
        teo_ratio,
        rms / context_rms,
        percentile_span,
        skewness,
        kurt,
        entropy,
    ], dtype=float)


def event_iou(a, b):
    a_onset = a.get("onset", a.get("start", 0))
    a_end = a.get("end_time", a.get("end", 0))
    b_onset = b.get("onset", b.get("start", 0))
    b_end = b.get("end_time", b.get("end", 0))
    overlap = max(0.0, min(a_end, b_end) - max(a_onset, b_onset))
    union = max(a_end, b_end) - min(a_onset, b_onset)
    return overlap / union if union > 0 else 0.0


def label_candidates(candidates, expert_events, iou_threshold=0.2):
    labels = []
    for candidate in candidates:
        best = 0.0
        for expert in expert_events:
            best = max(best, event_iou(candidate, expert))
        labels.append(1 if best >= iou_threshold else 0)
    return np.asarray(labels, dtype=int)
