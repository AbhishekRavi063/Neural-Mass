import numpy as np
from scipy.signal import welch

def calculate_spectral_snr(signal, sampling_frequency=100, signal_band=None):
    """
    Estimate SNR from the power spectrum.

    EEG-like oscillations are usually centered near zero, so mean/std SNR is
    misleading. This metric compares power around the dominant oscillatory
    component, or a provided frequency band, against the remaining broadband
    power. Higher is better.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")
    if len(signal) < 4 or np.std(signal) == 0:
        return 0.0

    centered = signal - np.mean(signal)
    nperseg = min(256, len(centered))
    frequencies, power = welch(centered, fs=sampling_frequency, nperseg=nperseg)

    valid = frequencies > 0
    frequencies = frequencies[valid]
    power = power[valid]
    if len(power) == 0 or np.sum(power) == 0:
        return 0.0

    if signal_band is None:
        dominant_frequency = frequencies[np.argmax(power)]
        half_width = max(1.0, sampling_frequency / nperseg)
        band_mask = np.abs(frequencies - dominant_frequency) <= half_width
    else:
        low, high = signal_band
        band_mask = (frequencies >= low) & (frequencies <= high)

    signal_power = np.sum(power[band_mask])
    noise_power = np.sum(power[~band_mask])
    if signal_power <= 0:
        return 0.0
    if noise_power <= 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)

def calculate_snr(signal, sampling_frequency=100, signal_band=None):
    """Backward-compatible alias for the spectral SNR metric."""
    return calculate_spectral_snr(signal, sampling_frequency, signal_band)

def calculate_rhythmicity(signal):
    """
    Calculates how steady the 'heartbeat' of the brain is.
    Uses autocorrelation to see if the signal repeats predictably.
    1.0 = Perfect rhythm, 0.0 = Pure chaos.
    Uses FFT-based autocorrelation for O(n log n) performance.
    """
    signal = np.asarray(signal)
    if len(signal) < 2:
        return 0.0
    sig = signal - np.mean(signal)
    if np.std(sig) == 0:
        return 0.0
    n = len(sig)
    # Zero-pad to 2n to avoid circular correlation artefacts
    fft_sig = np.fft.rfft(sig, n=2 * n)
    acf = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n].real
    if acf[0] == 0:
        return 0.0
    norm_corr = acf / acf[0]
    return float(np.max(norm_corr[int(len(norm_corr) * 0.1):]))

def calculate_rmse(signal, target):
    """Calculates the Root Mean Square Error between two signals."""
    signal = np.asarray(signal)
    target = np.asarray(target)
    if len(signal) != len(target):
        raise ValueError("signal and target must have the same length.")
    return np.sqrt(np.mean((signal - target)**2))

def calculate_correlation(signal, target):
    """Calculates the Pearson Correlation (Similarity) between two signals."""
    signal = np.asarray(signal)
    target = np.asarray(target)
    if len(signal) != len(target): return 0
    if np.std(signal) == 0 or np.std(target) == 0: return 0
    return np.corrcoef(signal, target)[0, 1]

def get_performance_report(signal, target=None, sampling_frequency=100):
    """Returns a dictionary of quality metrics."""
    metrics = {
        "SNR (dB)": round(calculate_snr(signal, sampling_frequency), 2),
        "Rhythmicity (0-1)": round(calculate_rhythmicity(signal), 2)
    }
    if target is not None:
        metrics["RMSE"] = round(calculate_rmse(signal, target), 4)
        metrics["Similarity"] = round(calculate_correlation(signal, target), 4)
    return metrics
