from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt, welch
from scipy.ndimage import uniform_filter1d
import numpy as np

def merge_close_events(mask: NDArray, max_gap: int) -> NDArray:
    """Merge boolean event regions separated by short gaps."""
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or not mask.any():
        return mask

    true_idxs = np.flatnonzero(mask)
    start = true_idxs[0]
    previous = true_idxs[0]
    for idx in true_idxs[1:]:
        gap = idx - previous - 1
        if gap <= max_gap:
            mask[previous + 1:idx] = True
        else:
            start = idx
        previous = idx
    return mask

def expand_events(mask: NDArray, padding: int) -> NDArray:
    """Expand boolean event regions by a fixed number of samples on both sides."""
    mask = np.asarray(mask, dtype=bool)
    if padding <= 0 or not mask.any():
        return mask.copy()

    expanded = mask.copy()
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    for start, end in zip(starts, ends):
        left = max(0, start - padding)
        right = min(len(mask), end + padding + 1)
        expanded[left:right] = True
    return expanded

def mask_segments(mask: NDArray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    return list(zip(starts, ends))

def _rise_time_ms(segment: NDArray, sampling_frequency: int) -> float:
    """Return the rise time in ms of the dominant deflection in `segment`.

    Uses the larger of the negative / positive excursion.  The rise time is
    measured as the time from the last zero-crossing *before* the peak to the
    peak itself.  Sharp artifact spikes have rise times < 50 ms; real
    K-complexes are ≥ 80 ms.
    """
    if len(segment) < 4:
        return 0.0
    neg_amp = abs(float(np.min(segment)))
    pos_amp = abs(float(np.max(segment)))
    if neg_amp >= pos_amp:
        peak_idx = int(np.argmin(segment))
        # Walk back from peak to last zero-crossing (signal ≥ 0)
        zc_idx = 0
        for i in range(peak_idx - 1, -1, -1):
            if segment[i] >= 0.0:
                zc_idx = i
                break
    else:
        peak_idx = int(np.argmax(segment))
        zc_idx = 0
        for i in range(peak_idx - 1, -1, -1):
            if segment[i] <= 0.0:
                zc_idx = i
                break
    rise_samples = peak_idx - zc_idx
    return rise_samples / sampling_frequency * 1000.0


def robust_std(x: NDArray) -> float:
    """Robust standard deviation estimator using Median Absolute Deviation (MAD)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad) if mad > 0 else float(np.std(x))


def _build_ideal_kc() -> NDArray:
    t = np.linspace(0, 1, 100)
    kc = np.zeros(100)
    kc[t < 0.35] = -np.sin(np.pi * t[t < 0.35] / 0.35)
    idx_pos = (t >= 0.35) & (t < 0.75)
    kc[idx_pos] = 0.6 * np.sin(np.pi * (t[idx_pos] - 0.35) / 0.40)
    idx_decay = t >= 0.75
    kc[idx_decay] = 0.6 * np.cos(np.pi * (t[idx_decay] - 0.75) / 0.50)
    return (kc - kc.mean()) / (kc.std() + 1e-10)

_IDEAL_KC: NDArray = _build_ideal_kc()


def filter_k_complex_candidates(
    mask: NDArray,
    filtered_signal: NDArray,
    sampling_frequency: int,
    min_event_duration: float,
    max_event_duration: float,
    min_peak_to_peak: float | NDArray,
    require_biphasic: bool,
    min_rise_time_ms: float = 0.0,
    max_rise_time_ms: float = 600.0,
    min_inter_event_gap: float = 0.5,
    hf_rms: NDArray | None = None,
    hf_artifact_ratio: float = 3.0,
    raw_signal: NDArray | None = None,
) -> NDArray:
    out = np.zeros_like(mask, dtype=bool)

    # Scale-invariant gating thresholds based on min_peak_to_peak
    if isinstance(min_peak_to_peak, np.ndarray):
        pos_gate_arr = 0.25 * min_peak_to_peak
        neg_gate_arr = -0.35 * min_peak_to_peak
    else:
        pos_gate_arr = np.full_like(filtered_signal, 0.25 * min_peak_to_peak)
        neg_gate_arr = np.full_like(filtered_signal, -0.35 * min_peak_to_peak)

    # Queue-like list to handle splitting of chained events
    segments = mask_segments(mask)
    verified_segments = []
    idx = 0
    while idx < len(segments):
        start, end = segments[idx]
        idx += 1
        orig_end = end

        segment = filtered_signal[start:end + 1]
        if len(segment) == 0:
            continue

        # Locate negative peak (primary anchor for K-complex)
        neg_idx = int(np.argmin(segment))
        abs_neg_idx = start + neg_idx
        neg_val = filtered_signal[abs_neg_idx]

        if require_biphasic:
            # Look forward from the negative peak (up to 1.2s) for the positive peak
            lookforward_samples = int(sampling_frequency * 1.2)
            lookforward_end = min(len(filtered_signal), abs_neg_idx + lookforward_samples)
            lookforward_seg = filtered_signal[abs_neg_idx:lookforward_end]

            if len(lookforward_seg) > 0:
                pos_idx_in_lookforward = int(np.argmax(lookforward_seg))
                abs_pos_idx = abs_neg_idx + pos_idx_in_lookforward
                pos_val = filtered_signal[abs_pos_idx]

                # Check that positive peak is significant and succeeds negative peak
                if abs_pos_idx > abs_neg_idx and pos_val >= pos_gate_arr[abs_pos_idx] and neg_val <= neg_gate_arr[abs_neg_idx]:
                    # Trim start: zero crossing before negative peak (up to 1.0s before)
                    prefix_start = max(0, abs_neg_idx - int(sampling_frequency * 1.0))
                    prefix = filtered_signal[prefix_start:abs_neg_idx]
                    if len(prefix) > 0:
                        pos_crossings = np.flatnonzero(prefix >= 0)
                        if len(pos_crossings) > 0:
                            start = max(0, prefix_start + pos_crossings[-1] + 1)
                        else:
                            start = prefix_start
                    else:
                        start = abs_neg_idx

                    # Trim end: zero crossing after positive peak (up to 1.0s after)
                    suffix_end = min(len(filtered_signal), abs_pos_idx + int(sampling_frequency * 1.0))
                    suffix = filtered_signal[abs_pos_idx:suffix_end]
                    if len(suffix) > 0:
                        neg_crossings = np.flatnonzero(suffix <= 0)
                        if len(neg_crossings) > 0:
                            end = abs_pos_idx + neg_crossings[0]
                        else:
                            end = suffix_end - 1
                    else:
                        end = abs_pos_idx
                else:
                    # Reject if peaks don't meet gating criteria
                    continue
            else:
                continue
        else:
            # Polarity-invariant trimming for non-biphasic / inverted cases
            segment = filtered_signal[start:end + 1]
            if len(segment) > 0:
                neg_idx_val = int(np.argmin(segment))
                pos_idx_val = int(np.argmax(segment))
                
                # Trim start to zero crossing before first peak
                first_peak_idx = min(neg_idx_val, pos_idx_val)
                zc_start_idx = 0
                first_peak_val = segment[first_peak_idx]
                if first_peak_val < 0:
                    for i in range(first_peak_idx - 1, -1, -1):
                        if segment[i] >= 0:
                            zc_start_idx = i
                            break
                else:
                    for i in range(first_peak_idx - 1, -1, -1):
                        if segment[i] <= 0:
                            zc_start_idx = i
                            break
                start = start + zc_start_idx + 1
                
                # Re-evaluate segment and peaks after adjusting start
                segment = filtered_signal[start:end + 1]
                if len(segment) > 0:
                    neg_idx_val = int(np.argmin(segment))
                    pos_idx_val = int(np.argmax(segment))
                    second_peak_idx = max(neg_idx_val, pos_idx_val)
                    
                    # Walk forward from second_peak_idx to find zero crossing
                    zc_end_idx = len(segment) - 1
                    second_peak_val = segment[second_peak_idx]
                    if second_peak_val < 0:
                        for i in range(second_peak_idx + 1, len(segment)):
                            if segment[i] >= 0:
                                zc_end_idx = i
                                break
                    else:
                        for i in range(second_peak_idx + 1, len(segment)):
                            if segment[i] <= 0:
                                zc_end_idx = i
                                break
                    end = start + zc_end_idx

        # Duration check on trimmed segment
        duration = (end - start + 1) / sampling_frequency
        if duration < min_event_duration or duration > max_event_duration:
            advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
            remaining_samples = orig_end - advance + 1
            if remaining_samples >= int(sampling_frequency * min_event_duration):
                segments.append((advance, orig_end))
            continue

        # Peak-to-peak amplitude check
        trimmed_segment = filtered_signal[start:end + 1]
        if len(trimmed_segment) == 0:
            continue
        peak_to_peak = float(np.max(trimmed_segment) - np.min(trimmed_segment))
        
        # Use local amplitude requirement if min_peak_to_peak is an array
        if isinstance(min_peak_to_peak, np.ndarray):
            local_min_ptp = float(min_peak_to_peak[abs_neg_idx])
        else:
            local_min_ptp = float(min_peak_to_peak)
            
        if peak_to_peak < local_min_ptp:
            advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
            remaining_samples = orig_end - advance + 1
            if remaining_samples >= int(sampling_frequency * min_event_duration):
                segments.append((advance, orig_end))
            continue

        # Rise-time filter (Sharpness)
        if min_rise_time_ms > 0.0 or max_rise_time_ms < float('inf'):
            rt = _rise_time_ms(trimmed_segment, sampling_frequency)
            if rt < min_rise_time_ms or rt > max_rise_time_ms:
                advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
                remaining_samples = orig_end - advance + 1
                if remaining_samples >= int(sampling_frequency * min_event_duration):
                    segments.append((advance, orig_end))
                continue

        # High-frequency artifact rejection
        if hf_rms is not None:
            # We measure the maximum HF RMS inside the wave compared to the surrounding background
            local_hf = np.max(hf_rms[start:end+1])
            
            # Estimate background HF RMS by looking up to 5 seconds before the event
            bg_start = max(0, start - int(sampling_frequency * 5.0))
            bg_end = max(1, start) # Prevent empty slice
            
            # Calculate background HF and enforce a global floor (median HF) to prevent ratio explosion on pure synthetic signals
            global_hf_median = np.median(hf_rms) if len(hf_rms) > 0 else 1e-6
            bg_hf = np.median(hf_rms[bg_start:bg_end]) if start > 0 else global_hf_median
            bg_hf = max(bg_hf, global_hf_median, 1e-6)
            
            if (local_hf / bg_hf) > hf_artifact_ratio:
                # The wave is contaminated by severe high-frequency noise (likely movement/arousal artifact)
                advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
                remaining_samples = orig_end - advance + 1
                if remaining_samples >= int(sampling_frequency * min_event_duration):
                    segments.append((advance, orig_end))
                continue

        # Refined ratio checks:
        if require_biphasic:
            min_val = np.min(trimmed_segment)
            max_val = np.max(trimmed_segment)
            
            # Both positive and negative peaks must be at least somewhat prominent (K-complexes are biphasic)
            # This helps reject pure monophasic artifacts like certain eye blinks
            if abs(min_val) < 0.25 * abs(max_val) or abs(max_val) < 0.25 * abs(min_val):
                advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
                remaining_samples = orig_end - advance + 1
                if remaining_samples >= int(sampling_frequency * min_event_duration):
                    segments.append((advance, orig_end))
                continue
                
            # Reject overly complex waveforms (e.g., trains of alpha/theta in Wake or EMG noise)
            # A true K-complex is a simple slow wave, meaning it crosses its mean very few times.
            # ZCR on the bandpassed signal, not raw — spindles riding on K-complexes
            # inflate raw ZCR and cause valid N2 events to be falsely rejected.
            band_trimmed = filtered_signal[start:end]
            mean_val = np.mean(band_trimmed)
            zero_crossings = np.sum(np.diff(np.sign(band_trimmed - mean_val)) != 0)
            duration_sec = len(band_trimmed) / sampling_frequency
            if duration_sec > 0 and zero_crossings / duration_sec > 3.0:
                continue
                
            # Positive-rebound check: positive peak must be at least 0.35 times the negative peak amplitude
            if abs(max_val) < 0.35 * abs(min_val):
                advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
                remaining_samples = orig_end - advance + 1
                if remaining_samples >= int(sampling_frequency * min_event_duration):
                    segments.append((advance, orig_end))
                continue

        # Template correlation gate: reject waveforms that don't match K-complex morphology.
        # Catches Wake slow waves and eye blinks that pass the biphasic amplitude checks.
        if require_biphasic and len(trimmed_segment) >= 8:
            interp_seg = np.interp(
                np.linspace(0, len(trimmed_segment) - 1, 100),
                np.arange(len(trimmed_segment)),
                trimmed_segment,
            )
            interp_seg = (interp_seg - np.mean(interp_seg)) / (np.std(interp_seg) + 1e-10)
            if float(np.corrcoef(interp_seg, _IDEAL_KC)[0, 1]) < 0.15:
                advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
                remaining_samples = orig_end - advance + 1
                if remaining_samples >= int(sampling_frequency * min_event_duration):
                    segments.append((advance, orig_end))
                continue

        # Alpha-context gate: reject candidates where the surrounding 40-second window
        # has elevated alpha/delta ratio — proxy for Wakefulness or N1.
        if raw_signal is not None:
            ctx_r = int(sampling_frequency * 20)
            ctx = raw_signal[max(0, abs_neg_idx - ctx_r): min(len(raw_signal), abs_neg_idx + ctx_r)]
            if len(ctx) >= int(sampling_frequency * 4):
                lfreqs, lpsd = welch(ctx, fs=float(sampling_frequency), nperseg=min(512, len(ctx)))
                delta_pwr = float(np.sum(lpsd[(lfreqs >= 0.5) & (lfreqs <= 4.0)])) + 1e-10
                alpha_pwr = float(np.sum(lpsd[(lfreqs >= 8.0) & (lfreqs <= 13.0)]))
                if alpha_pwr / delta_pwr > 2.5:
                    advance = max(end + 1, abs_neg_idx + int(sampling_frequency * 0.1))
                    remaining_samples = orig_end - advance + 1
                    if remaining_samples >= int(sampling_frequency * min_event_duration):
                        segments.append((advance, orig_end))
                    continue

        verified_segments.append((start, end))

        # Requeue remaining
        remaining_samples = orig_end - end
        if remaining_samples >= int(sampling_frequency * min_event_duration):
            segments.append((end + 1, orig_end))

    if not verified_segments:
        return out

    # Sort segments by start sample
    verified_segments.sort(key=lambda x: x[0])

    # Enforce inter-event gap
    gap_samples = int(sampling_frequency * min_inter_event_gap)
    kept = []
    for s, e in verified_segments:
        if not kept:
            kept.append((s, e))
            continue
        prev_s, prev_e = kept[-1]
        prev_neg = prev_s + np.argmin(filtered_signal[prev_s:prev_e+1])
        cur_neg = s + np.argmin(filtered_signal[s:e+1])

        if cur_neg - prev_neg < gap_samples:
            ptp_cur  = float(np.max(filtered_signal[s:e+1]) - np.min(filtered_signal[s:e+1]))
            ptp_prev = float(np.max(filtered_signal[prev_s:prev_e+1]) - np.min(filtered_signal[prev_s:prev_e+1]))
            if ptp_cur > ptp_prev:
                kept[-1] = (s, e)
        else:
            min_boundary_gap = int(sampling_frequency * 0.15)
            if s - prev_e - 1 < min_boundary_gap:
                mid = (prev_e + s) // 2
                new_prev_e = max(prev_s, mid - min_boundary_gap // 2)
                new_s = min(e, mid + (min_boundary_gap + 1) // 2)
                if new_prev_e >= prev_s and new_s <= e:
                    kept[-1] = (prev_s, new_prev_e)
                    s = new_s
            kept.append((s, e))

    for s, e in kept:
        out[s:e+1] = True

    return out




def spindle_detection(signal : NDArray,
                      sampling_frequency : int = 300,
                      threshold_std : float = 1.5,
                      min_duration : float = 0.5) -> NDArray:
    """Function to detect spindles

    ARGS:
    -----
        signal : The input signal as a 1D NDarray
        sampling_frequency : The sampling frequency of the signal the default is 300 point per second

    RETURNS:
    ------
        A mask with TRUE where there is a spindle
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or Inf. Remove artifacts before detection.")
    window_size = int(sampling_frequency * 0.2)
    if len(signal) < window_size:
        return np.zeros(len(signal), dtype=bool)

    #fft_values = fft(signal)
    #frequencies = fftfreq(len(signal))

    # Bandpass the signal. Order 4 is standard for EEG — order 10 risks
    # ringing and instability with sosfiltfilt on short segments.
    sos = butter(4, [11, 16],
                 btype="bandpass",
                 output="sos",
                 fs=sampling_frequency)
    filtered_signal = sosfiltfilt(sos, signal)

    # Root mean square over sliding windows
    windows = sliding_window_view(filtered_signal,
                                  window_shape=window_size)
    windows = windows**2
    windows = np.sqrt(np.mean(windows,axis=1))
    
    # Mark windows whose sigma-band RMS is clearly above the local background.
    std_signal = np.std(signal) or 1.0
    std_filtered = np.std(filtered_signal)
    std_windows = np.std(windows)
    median_windows = np.median(windows)
    max_windows = np.max(windows) if len(windows) > 0 else 0.0

    is_continuous_spindle = (
        (std_windows < 0.20 * median_windows or median_windows > 0.35 * max_windows) and
        std_filtered > 0.10 * std_signal and
        median_windows > 1e-5
    )

    if is_continuous_spindle:
        mask_out = windows >= 0.8 * median_windows
    else:
        threshold = median_windows + threshold_std * std_windows
        mask_out = windows > threshold

    # Filter runs by minimum duration using numpy run-length encoding.
    min_len = int(sampling_frequency * min_duration)
    padded = np.concatenate(([False], mask_out, [False]))
    diff = np.diff(padded.view(np.int8))
    run_starts = np.flatnonzero(diff == 1)
    run_ends = np.flatnonzero(diff == -1)   # exclusive end in mask_out coords
    true_mask = np.zeros(len(signal), dtype=bool)
    for start, end in zip(run_starts, run_ends):
        if end - start >= min_len:
            sample_end = min(len(signal), end + window_size - 1)
            true_mask[start:sample_end] = True

    return true_mask

def splindle_detection(signal : NDArray,
                      sampling_frequency : int = 300) -> NDArray:
    """Backward-compatible alias for the original misspelled function name."""
    return spindle_detection(signal, sampling_frequency)

def K_complex_detection(signal : NDArray,
                        sampling_frequency : int = 1000,
                        threshold_std : float = 2.5,
                        min_duration : float = 0.2,
                        merge_gap : float = 0.35,
                        event_padding : float = 0.18,
                        min_event_duration : float = 0.45,
                        max_event_duration : float = 2.0,
                        min_peak_to_peak : float | None = None,
                        absolute_min_ptp : float | None = None,
                        require_biphasic : bool = True,
                        min_rise_time_ms : float = 80.0,
                        max_rise_time_ms : float = 600.0,
                        min_inter_event_gap : float = 0.5,
                        reject_artifacts : bool = True) -> NDArray:
    """
    Function that returns a mask where a K-Complex occurs.
    K-complexes are large, slow waves (>0.5s duration, high amplitude).

    Parameters
    ----------
    min_peak_to_peak : float or None
        Minimum peak-to-peak amplitude required for a K-complex candidate.
        When None (default), computed adaptively as 1.5 * std of the bandpass-
        filtered signal, making the detector unit-agnostic (works on both µV
        real EEG and arbitrary-unit synthetic signals without manual tuning).
        Pass an explicit float to restore fixed-threshold behaviour.
    min_rise_time_ms : float
        Minimum rise time in milliseconds for the dominant deflection.
        real K-complexes develop over >= 80 ms; spike artifacts (EMG, movement)
        rise in < 50 ms. Set to 0 to disable. Default: 80 ms.
    max_rise_time_ms : float
        Maximum rise time in milliseconds. Slow, rounded delta waves take too long
        to rise and lack the sharpness of K-complexes. Default: 600 ms.
    min_inter_event_gap : float
        Minimum required gap in seconds between two successive detected events.
        When two events are closer than this, only the larger-amplitude one is
        kept. Prevents double-detection of the same K-complex. Default: 0.5 s.
    reject_artifacts : bool
        If True, rejects candidates that co-occur with a massive burst of
        high-frequency energy (e.g., >15 Hz muscle twitches/movement).
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or Inf. Remove artifacts before detection.")
    if len(signal) == 0:
        return np.zeros(0, dtype=bool)

    # 1. Isolate large slow waves while removing very slow baseline drift.
    print("Step 1: bandpass filter...", flush=True)
    sos = butter(4, [0.5, 2.5],
                 btype="bandpass",
                 output="sos",
                 fs=sampling_frequency)
    filtered_signal = sosfiltfilt(sos, signal)

    # 2. Adaptive amplitude floor & Threshold via Rolling Baseline
    print("Step 2: rolling baseline...", flush=True)
    # Use a 5-minute rolling window to compute local standard deviation.
    # We use the absolute value smoothed to approximate the envelope standard deviation.
    window_samples = int(sampling_frequency * 5 * 60)
    # std(x) for zero-mean normal approx is sqrt(pi/2) * mean(|x|)
    rolling_std = uniform_filter1d(np.abs(filtered_signal), size=window_samples) * np.sqrt(np.pi / 2.0)
    
    # Prevent threshold from collapsing to 0 during long wake periods
    global_std = robust_std(filtered_signal)
    rolling_std = np.maximum(rolling_std, global_std * 0.5)
    
    if absolute_min_ptp is not None:
        # Floor the rolling std so we don't pick up millions of tiny noise peaks
        rolling_std = np.maximum(rolling_std, absolute_min_ptp / 4.0)
    
    # Fallback if signal is too short
    if len(filtered_signal) < window_samples // 10:
        rolling_std = np.full_like(filtered_signal, global_std)
        
    adaptive_mode = min_peak_to_peak is None
    if adaptive_mode:
        min_peak_to_peak = 1.5 * rolling_std
        if absolute_min_ptp is not None:
            min_peak_to_peak = np.maximum(min_peak_to_peak, absolute_min_ptp)
        threshold_std = min(threshold_std, 1.5)
        min_duration = min(min_duration, 0.10)
    else:
        # If static float, maintain it
        min_peak_to_peak = np.full_like(filtered_signal, float(min_peak_to_peak))

    # 3. Detect peaks that exceed a high amplitude threshold.
    threshold = threshold_std * rolling_std
    mask_out = np.abs(filtered_signal) > threshold
    
    # Optional Artifact Rejection Prep
    hf_rms = None
    if reject_artifacts:
        print("Step 3: high frequency RMS...", flush=True)
        hf_sos = butter(4, 15, btype="highpass", output="sos", fs=sampling_frequency)
        hf_sig = sosfiltfilt(hf_sos, signal)
        hf_variance = uniform_filter1d(hf_sig**2, size=int(sampling_frequency * 0.5))
        hf_rms = np.sqrt(np.clip(hf_variance, 0.0, None))
    
    # 3. Filter for minimum duration using numpy run-length encoding.
    print("Step 4: thresholding and masking...", flush=True)
    min_len = int(sampling_frequency * min_duration)
    padded = np.concatenate(([False], mask_out, [False]))
    diff = np.diff(padded.view(np.int8))
    run_starts = np.flatnonzero(diff == 1)
    run_ends = np.flatnonzero(diff == -1)
    true_mask = np.zeros(len(mask_out), dtype=bool)
    for start, end in zip(run_starts, run_ends):
        if end - start >= min_len:
            true_mask[start:end] = True

    print("Step 5: merge and expand...", flush=True)
    merged = merge_close_events(true_mask, int(sampling_frequency * merge_gap))
    expanded = expand_events(merged, int(sampling_frequency * event_padding))
    
    print("Step 6: filtering candidates...", flush=True)
    shaped = filter_k_complex_candidates(
        expanded,
        filtered_signal,
        sampling_frequency,
        min_event_duration,
        max_event_duration,
        min_peak_to_peak,
        require_biphasic,
        min_rise_time_ms=min_rise_time_ms,
        max_rise_time_ms=max_rise_time_ms,
        min_inter_event_gap=min_inter_event_gap,
        hf_rms=hf_rms,
        raw_signal=signal,
    )

    return shaped
