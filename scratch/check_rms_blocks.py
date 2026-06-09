import numpy as np
from neural_mass import ThalamocorticalSleepModel
from scipy.signal import butter, sosfiltfilt

model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
signals = model.simulate(seconds=180.0, sampling_frequency=200)
eeg = signals["eeg"]

sos = butter(4, [11, 16], btype="bandpass", output="sos", fs=200)
filtered = sosfiltfilt(sos, eeg)

window_size = int(200 * 0.2)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(filtered, window_shape=window_size)
windows = np.sqrt(np.mean(windows**2, axis=1))

print("Block-wise RMS analysis (every 10s):")
block_size = 200 * 10
for i in range(18):
    start_idx = i * block_size
    end_idx = min(len(filtered), (i + 1) * block_size)
    block_rms = np.std(filtered[start_idx:end_idx])
    print(f"  Block {i+1} (time {i*10}-{(i+1)*10}s): Filtered Std = {block_rms:.6f}")
