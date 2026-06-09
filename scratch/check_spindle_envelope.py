import numpy as np
from neural_mass import ThalamocorticalSleepModel
from scipy.signal import butter, sosfiltfilt

model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
signals = model.simulate(seconds=30.0, sampling_frequency=200)
eeg = signals["eeg"]
spindle = signals["spindle"]

# Bandpass filter eeg in spindle band
sos = butter(4, [11, 16], btype="bandpass", output="sos", fs=200)
filtered = sosfiltfilt(sos, eeg)

# Sliding RMS
window_size = int(200 * 0.2)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(filtered, window_shape=window_size)
windows = np.sqrt(np.mean(windows**2, axis=1))

print("Windows RMS shape:", windows.shape)
print("Min RMS:", np.min(windows))
print("Max RMS:", np.max(windows))
print("Mean RMS:", np.mean(windows))
print("Std RMS:", np.std(windows))
print("Median RMS:", np.median(windows))

# Print first 20 window values
print("First 50 RMS values:", windows[:50])
