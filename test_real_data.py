import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import generate_realistic_eeg
from src.metrics import get_performance_report
from src.event_detection import K_complex_detection

def count_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(np.sum(starts))

# 1. GENERATE REALISTIC MOCK EEG (Faster than downloading for this demo)
print("Generating realistic human-like EEG data...")
real_wave, event_masks = generate_realistic_eeg(seed=42, return_events=True)
sfreq = 100

# 2. RUN PERFORMANCE METRICS
print("\n--- PERFORMANCE METRICS (REALISTIC SIGNAL) ---")
report = get_performance_report(real_wave)
print(f"Signal Scores: {report}")

# 3. RUN EVENT DETECTION
print("\n--- DETECTING K-COMPLEXES IN SIGNAL ---")
k_complexes = K_complex_detection(real_wave, sampling_frequency=sfreq)
print(f"K-complex events injected: {count_segments(event_masks['k_complex'])}")
print(f"K-complex events detected: {count_segments(k_complexes)}")

# 4. VISUALIZE
plt.figure(figsize=(12, 4))
plt.plot(real_wave, color='#2980b9')
plt.fill_between(
    np.arange(len(real_wave)),
    np.min(real_wave),
    np.max(real_wave),
    where=event_masks["k_complex"],
    color="#9bdbff",
    alpha=0.35,
    label="Injected K-complex",
)
plt.fill_between(
    np.arange(len(real_wave)),
    np.min(real_wave),
    np.max(real_wave),
    where=k_complexes,
    color="#ffb3d9",
    alpha=0.35,
    label="Detected K-complex",
)
plt.title("Realistic Human-like EEG (Simulated)")
plt.xlabel("Time steps")
plt.ylabel("Potential (uV)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.2)
plt.savefig('real_data_test.png')
print("\nTest complete! Image saved to 'real_data_test.png'")
