"""EEG preprocessing utilities: filtering, artifact rejection, and rescaling.

This module provides a lightweight preprocessing pipeline that should be applied
to raw EEG signals *before* passing them to any detector or fitting function.
It does NOT require MNE or EEGLAB — only NumPy and SciPy.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt_zi


def bandpass_filter(
    signal: NDArray,
    sfreq: float,
    low: float = 0.3,
    high: float = 40.0,
    order: int = 4,
) -> NDArray[np.float64]:
    """Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : 1-D array
    sfreq : float — sampling frequency in Hz
    low, high : float — passband edges in Hz
    order : int — filter order (default 4; higher values risk ringing on short signals)

    Returns
    -------
    Filtered signal, same shape as input.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D.")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains NaN or Inf — remove artifacts first.")
    nyq = sfreq / 2.0
    high = min(high, nyq * 0.99)
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, signal - np.median(signal))


def notch_filter(
    signal: NDArray,
    sfreq: float,
    freq: float = 50.0,
    quality: float = 30.0,
) -> NDArray[np.float64]:
    """Zero-phase IIR notch filter to remove power-line interference.

    Parameters
    ----------
    freq : float — notch frequency in Hz (50 Hz Europe / 60 Hz North America)
    quality : float — Q factor; higher = narrower notch
    """
    signal = np.asarray(signal, dtype=float)
    b, a = iirnotch(freq, quality, fs=sfreq)
    # Convert ba to sos for numerical stability
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, signal)


def amplitude_clip(
    signal: NDArray,
    threshold: float | None = None,
    n_std: float = 5.0,
) -> NDArray[np.float64]:
    """Clip amplitude spikes that exceed a threshold.

    Replaces clipped samples with NaN so downstream code can detect them.

    Parameters
    ----------
    threshold : float or None
        Absolute amplitude ceiling. When None, computed as ``n_std * std(signal)``.
    n_std : float
        Number of standard deviations for auto threshold (default 5.0).

    Returns
    -------
    Signal with outliers replaced by NaN.
    """
    signal = np.asarray(signal, dtype=float)
    if threshold is None:
        threshold = n_std * np.std(signal)
    clipped = signal.copy()
    clipped[np.abs(signal) > threshold] = np.nan
    return clipped


def interpolate_nan(signal: NDArray) -> NDArray[np.float64]:
    """Linear interpolation of NaN segments.

    Short artifact segments (created by amplitude_clip) are bridged by
    linear interpolation.  Segments longer than 10% of the signal are left
    as NaN so callers can decide how to handle them.
    """
    signal = np.asarray(signal, dtype=float)
    nan_mask = np.isnan(signal)
    if not nan_mask.any():
        return signal.copy()

    max_gap = int(0.1 * len(signal))
    out = signal.copy()
    good = np.flatnonzero(~nan_mask)
    if len(good) < 2:
        return out

    # Find contiguous NaN runs
    starts = np.flatnonzero(nan_mask & ~np.roll(nan_mask, 1))
    ends = np.flatnonzero(nan_mask & ~np.roll(nan_mask, -1))
    for s, e in zip(starts, ends):
        if e - s + 1 <= max_gap:
            left = s - 1 if s > 0 else None
            right = e + 1 if e < len(signal) - 1 else None
            if left is not None and right is not None:
                # out[s:e+1] needs e-s+1 values (anchors excluded).
                # linspace of length e-s+3 gives anchors at [0] and [-1],
                # so [1:-1] yields exactly e-s+1 interpolated values.
                out[s : e + 1] = np.linspace(out[left], out[right], e - s + 3)[1:-1]
    return out


def zscore_normalize(signal: NDArray) -> NDArray[np.float64]:
    """Z-score the signal (zero mean, unit variance)."""
    signal = np.asarray(signal, dtype=float)
    mu = np.nanmean(signal)
    sigma = np.nanstd(signal)
    if sigma < 1e-12:
        return signal - mu
    return (signal - mu) / sigma


def preprocess_eeg(
    signal: NDArray,
    sfreq: float,
    *,
    high_pass: float = 0.3,
    low_pass: float = 40.0,
    notch_freq: float | None = 50.0,
    notch_quality: float = 30.0,
    artifact_threshold_std: float = 5.0,
    interpolate_artifacts: bool = True,
) -> NDArray[np.float64]:
    """Full EEG preprocessing pipeline.

    Steps applied in order:
    1. Bandpass filter (high_pass – low_pass Hz)
    2. Optional notch filter at notch_freq Hz (power-line interference)
    3. Amplitude artifact clipping (samples > artifact_threshold_std σ → NaN)
    4. Optional linear interpolation of short artifact gaps

    Parameters
    ----------
    signal : 1-D array — raw EEG (any amplitude unit)
    sfreq : float — sampling frequency in Hz
    high_pass : float — high-pass cutoff in Hz (default 0.3)
    low_pass : float — low-pass cutoff in Hz (default 40)
    notch_freq : float or None — notch frequency; None to skip (default 50 Hz)
    notch_quality : float — Q factor for notch filter
    artifact_threshold_std : float — number of σ above which samples are artifacts
    interpolate_artifacts : bool — interpolate clipped segments (default True)

    Returns
    -------
    Preprocessed signal, shape (n_samples,)
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D.")

    # 1. Bandpass
    out = bandpass_filter(signal, sfreq, low=high_pass, high=low_pass)

    # 2. Notch
    if notch_freq is not None and notch_freq < sfreq / 2:
        out = notch_filter(out, sfreq, freq=notch_freq, quality=notch_quality)

    # 3. Artifact clip
    out = amplitude_clip(out, n_std=artifact_threshold_std)

    # 4. Interpolate
    if interpolate_artifacts:
        out = interpolate_nan(out)

    # If any NaN remain (long artifacts), replace with zeros for detector safety
    out = np.where(np.isnan(out), 0.0, out)

    return out


def adaptive_scale(
    signal: NDArray,
    target_std: float = 50.0,
) -> tuple[NDArray[np.float64], float]:
    """Scale signal so its std matches target_std.

    Useful for aligning synthetic (arbitrary-unit) EEG with real (µV) EEG
    before comparison or combined fitting.

    Returns
    -------
    scaled_signal : NDArray
    scale_factor : float — multiply original by this to get scaled version
    """
    signal = np.asarray(signal, dtype=float)
    sigma = np.std(signal)
    if sigma < 1e-12:
        return signal.copy(), 1.0
    factor = target_std / sigma
    return signal * factor, factor


def clinical_artifact_filter(
    signal: NDArray,
    sfreq: float,
    high_pass: float = 0.1,
    low_pass: float = 45.0,
    reject_std_threshold: float = 4.5,
) -> NDArray[np.float64]:
    """Clinically filter EEG and reject high-variance artifact epochs.

    1. Zero-phase Butterworth bandpass filter (0.1 - 45 Hz).
    2. Split signal into 2-second epochs.
    3. Rejects epochs where std exceeds reject_std_threshold * global_std.
    4. Interpolate rejected epochs linearly.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D.")

    # 1. Bandpass filter
    out = bandpass_filter(signal, sfreq, low=high_pass, high=low_pass, order=4)

    # 2. Divide into 2-second epochs
    epoch_len = int(round(2.0 * sfreq))
    n_epochs = len(out) // epoch_len

    global_std = np.std(out)
    if global_std < 1e-12:
        return out

    out_clean = out.copy()
    rejected_mask = np.zeros(len(out), dtype=bool)

    for epoch_idx in range(n_epochs):
        start = epoch_idx * epoch_len
        end = start + epoch_len
        epoch_data = out[start:end]

        if np.std(epoch_data) > reject_std_threshold * global_std:
            out_clean[start:end] = np.nan
            rejected_mask[start:end] = True

    # Handle remainder
    if len(out) % epoch_len > 0:
        start = n_epochs * epoch_len
        epoch_data = out[start:]
        if np.std(epoch_data) > reject_std_threshold * global_std:
            out_clean[start:] = np.nan
            rejected_mask[start:] = True

    # 3. Interpolate the NaN epochs using helper
    if rejected_mask.any():
        out_clean = interpolate_nan(out_clean)

    # Clean remaining NaNs
    out_clean = np.where(np.isnan(out_clean), 0.0, out_clean)

    return out_clean
