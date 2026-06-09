import numpy as np
from neural_mass import ThalamocorticalSleepModel
from scipy.signal import butter, sosfiltfilt
from numpy.lib.stride_tricks import sliding_window_view

def test_signal_detection(signal, sfreq):
    window_size = int(sfreq * 0.2)
    sos = butter(4, [11, 16], btype="bandpass", output="sos", fs=sfreq)
    filtered = sosfiltfilt(sos, signal)
    windows = sliding_window_view(filtered, window_shape=window_size)
    windows = windows**2
    windows = np.sqrt(np.mean(windows, axis=1))
    
    std_sig = np.std(signal) or 1.0
    std_fil = np.std(filtered)
    std_win = np.std(windows)
    med_win = np.median(windows)
    max_win = np.max(windows)
    
    is_cont = (
        (std_win < 0.20 * med_win or med_win > 0.35 * max_win) and
        std_fil > 0.10 * std_sig and
        med_win > 1e-5
    )
    
    if is_cont:
        thresh = 0.8 * med_win
    else:
        thresh = med_win + 1.5 * std_win
        
    above = np.sum(windows > thresh)
    return is_cont, thresh, max_win, above

# Test 1: Synthetic N2
model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
out = model.simulate(seconds=60.0, sampling_frequency=200)
is_cont1, thresh1, max1, above1 = test_signal_detection(out["eeg"], 200)
print("Synthetic N2:")
print(f"  is_cont: {is_cont1}")
print(f"  threshold: {thresh1:.6f}")
print(f"  max window: {max1:.6f}")
print(f"  windows above: {above1}")

# Test 2: Standard burst test signal
sampling_frequency = 100
times = np.arange(5 * sampling_frequency) / sampling_frequency
signal = 0.1 * np.sin(2 * np.pi * 2 * times)
burst_mask = (times >= 2.0) & (times < 3.0)
signal[burst_mask] += 5.0 * np.sin(2 * np.pi * 12 * times[burst_mask])
is_cont2, thresh2, max2, above2 = test_signal_detection(signal, 100)
print("Sustained Sigma Burst:")
print(f"  is_cont: {is_cont2}")
print(f"  threshold: {thresh2:.6f}")
print(f"  max window: {max2:.6f}")
print(f"  windows above: {above2}")
