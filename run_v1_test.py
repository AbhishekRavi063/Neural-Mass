import numpy as np
import matplotlib.pyplot as plt
from src.mass_model import MassModel
from src.event_detection import splindle_detection, K_complex_detection

# 1. SETUP: Create a 2-node network (Cortex and Thalamus)
# We will use the parameters we developed in the research phases
pop_configs = [
    {"A": 3.25, "B": 22.0}, # Population 0: Cortex
    {"A": 3.25, "B": 22.0}  # Population 1: Thalamus
]

# Connections: (source, target, weight)
# Cortex -> Thalamus (50.0) and Thalamus -> Cortex (50.0)
connections = [
    (0, 1, 50.0), 
    (1, 0, 50.0)
]

# 2. SIMULATE
print("Running v1 Library Simulation (3 seconds)...")
model = MassModel(pop_configs, connections)
signals = model.simulate(seconds=3.0)

# 3. ANALYZE (Event Detection)
# Let's check for spindles in the Cortex signal (Column 0)
cortex_eeg = signals[:, 0]
spindles = splindle_detection(cortex_eeg, sampling_frequency=1000)
k_complexes = K_complex_detection(cortex_eeg, sampling_frequency=1000)

print(f"Spindles detected: {np.any(spindles)}")
print(f"K-complexes detected: {np.any(k_complexes)}")

# 4. VISUALIZE
plt.figure(figsize=(12, 6))
plt.plot(cortex_eeg, color='#2c3e50', label='Cortex EEG')

# Highlight detected events
if np.any(spindles):
    plt.fill_between(range(len(cortex_eeg)), np.min(cortex_eeg), np.max(cortex_eeg), 
                     where=spindles, color='yellow', alpha=0.3, label='Spindle Detected')
    
plt.title("v1 Library Test: Simulation and Event Detection")
plt.xlabel("Time (ms)")
plt.ylabel("Potential (mV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('v1_library_test.png')
print("Test complete. Results saved to 'v1_library_test.png'")
# plt.show()
