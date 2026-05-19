from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt
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

def filter_k_complex_candidates(
    mask: NDArray,
    filtered_signal: NDArray,
    sampling_frequency: int,
    min_event_duration: float,
    max_event_duration: float,
    min_peak_to_peak: float,
    require_biphasic: bool,
) -> NDArray:
    out = np.zeros_like(mask, dtype=bool)

    for start, end in mask_segments(mask):
        duration = (end - start + 1) / sampling_frequency
        if duration < min_event_duration or duration > max_event_duration:
            continue

        segment = filtered_signal[start:end + 1]
        peak_to_peak = float(np.max(segment) - np.min(segment))
        if peak_to_peak < min_peak_to_peak:
            continue

        if require_biphasic:
            negative_idx = int(np.argmin(segment))
            positive_idx = int(np.argmax(segment))
            if negative_idx >= positive_idx:
                continue
            if abs(np.min(segment)) < 0.5 * abs(np.max(segment)):
                continue

        out[start:end + 1] = True

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
    window_size = int(sampling_frequency * 0.2)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if len(signal) < window_size:
        return np.zeros(len(signal), dtype=bool)

    #fft_values = fft(signal)
    #frequencies = fftfreq(len(signal))

    # Bandpass the signal
    sos = butter(10, [11,16],
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
    threshold = np.median(windows) + threshold_std * np.std(windows)
    mask_out = windows>threshold

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
                        max_event_duration : float = 1.8,
                        min_peak_to_peak : float = 50.0,
                        require_biphasic : bool = True) -> NDArray:
    """
    Function that returns a mask where a K-Complex occurs.
    K-complexes are large, slow waves (>0.5s duration, high amplitude).
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if len(signal) == 0:
        return np.zeros(0, dtype=bool)

    # 1. Isolate large slow waves while removing very slow baseline drift.
    sos = butter(4, [0.5, 5],
                 btype="bandpass",
                 output="sos",
                 fs=sampling_frequency)
    filtered_signal = sosfiltfilt(sos, signal)
    
    # 2. Detect peaks that exceed a high amplitude threshold.
    threshold = threshold_std * np.std(filtered_signal)
    
    mask_out = np.abs(filtered_signal) > threshold
    
    # 3. Filter for minimum duration using numpy run-length encoding.
    min_len = int(sampling_frequency * min_duration)
    padded = np.concatenate(([False], mask_out, [False]))
    diff = np.diff(padded.view(np.int8))
    run_starts = np.flatnonzero(diff == 1)
    run_ends = np.flatnonzero(diff == -1)
    true_mask = np.zeros(len(mask_out), dtype=bool)
    for start, end in zip(run_starts, run_ends):
        if end - start >= min_len:
            true_mask[start:end] = True

    merged = merge_close_events(true_mask, int(sampling_frequency * merge_gap))
    expanded = expand_events(merged, int(sampling_frequency * event_padding))
    return filter_k_complex_candidates(
        expanded,
        filtered_signal,
        sampling_frequency,
        min_event_duration,
        max_event_duration,
        min_peak_to_peak,
        require_biphasic,
    )
