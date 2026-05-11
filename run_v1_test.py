import numpy as np
import matplotlib.pyplot as plt
from src.graph import Population, Connection, ComputationalGraph
from src.event_detection import spindle_detection, K_complex_detection

# 1. SETUP: Create a 2-node graph (Cortex and Thalamus)
cortex = Population(N=1000)
thalamus = Population(N=1000)

# Directed edges: cortex <-> thalamus
# Lowering weights to 10.0 for stability
conns = [
    Connection(source=cortex, target=thalamus, weight=10.0),
    Connection(source=thalamus, target=cortex, weight=10.0)
]

# 2. INITIALIZE GRAPH
# Using a smaller dt (0.001) is critical for Jansen-Rit stability
graph = ComputationalGraph(
    populations=[cortex, thalamus],
    connections=conns,
    dt=0.001,
    seed=42,
)

# 3. SIMULATE
print("Running Graph-Based Simulation (1 second)...")
signals = graph.simulate(seconds=1.0)

# 4. ANALYZE (Cortex signal is at index 0)
cortex_eeg = signals[:, 0]
spindles = spindle_detection(cortex_eeg, sampling_frequency=100)
k_complexes = K_complex_detection(cortex_eeg, sampling_frequency=100)

print(f"Spindles detected: {np.any(spindles)}")
print(f"K-complexes detected: {np.any(k_complexes)}")

# 5. VISUALIZE
plt.figure(figsize=(12, 6))
plt.plot(cortex_eeg, color='#16a085', label='Cortex EEG (Graph Mode)')

plt.title("Graph-Based Architecture: Simulation Result")
plt.xlabel("Time steps")
plt.ylabel("Potential (mV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('graph_test_result.png')
print("Test complete. Results saved to 'graph_test_result.png'")
