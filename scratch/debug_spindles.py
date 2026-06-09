import numpy as np
from scipy.signal import welch
from neural_mass import ThalamocorticalSleepModel
from neural_mass.detection.event_detection import spindle_detection, mask_segments

model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
signals = model.simulate(seconds=180.0, sampling_frequency=200, multi_channel=True)
eeg = signals["eeg"]
spindle_state = signals["spindle"]

print("EEG Std:", np.std(eeg))
print("Spindle State Std:", np.std(spindle_state))

# Let's inspect the bandpass filtered EEG in the spindle band (11-16 Hz)
from scipy.signal import butter, sosfiltfilt
sos = butter(4, [11, 16], btype="bandpass", output="sos", fs=200)
filtered_eeg = sosfiltfilt(sos, eeg)
print("Filtered EEG (11-16 Hz) Std:", np.std(filtered_eeg))

# Let's run spindle detection step by step
sampling_frequency = 200
threshold_std = 1.5
window_size = int(sampling_frequency * 0.2)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(filtered_eeg, window_shape=window_size)
windows = np.sqrt(np.mean(windows**2, axis=1))

threshold = np.median(windows) + threshold_std * np.std(windows)
print("Median of windows:", np.median(windows))
print("Std of windows:", np.std(windows))
print("Threshold:", threshold)
print("Max window value:", np.max(windows))
print("Number of windows above threshold:", np.sum(windows > threshold))

mask = spindle_detection(eeg, sampling_frequency=200)
print("Detected spindles count:", len(mask_segments(mask)))
