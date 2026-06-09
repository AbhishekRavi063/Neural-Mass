import numpy as np
from neural_mass.utils.preprocessing import clinical_artifact_filter

sfreq = 100.0
t = np.arange(10 * sfreq) / sfreq
signal = np.sin(2.0 * np.pi * 2.0 * t)

# Inject high-variance noise in Epoch 2 (seconds 4.0 to 6.0)
rng = np.random.default_rng(42)
signal[int(4.0 * sfreq) : int(6.0 * sfreq)] += rng.normal(0.0, 25.0, size=200)

filtered = clinical_artifact_filter(signal, sfreq=sfreq, reject_std_threshold=2.0)

print(f"Original signal max in epoch 2: {np.max(np.abs(signal[400:600])):.4f}")
print(f"Filtered signal max in epoch 2: {np.max(np.abs(filtered[400:600])):.4f}")
print(f"Filtered signal max overall: {np.max(np.abs(filtered)):.4f}")
print(f"Epoch 2 std: {np.std(signal[400:600]):.4f}")
print(f"Global std: {np.std(signal):.4f}")
print(f"Ratio: {np.std(signal[400:600]) / np.std(signal):.4f}")
