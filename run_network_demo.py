import numpy as np
import matplotlib.pyplot as plt
from src.graph import Population, Connection, ComputationalGraph

# 1. INITIALIZE BRAIN REGIONS
print("Initializing 4-Region Mini-Brain...")
frontal = Population(name="Frontal")
occipital = Population(name="Occipital")
thalamus = Population(name="Thalamus")
reticular = Population(name="Reticular") # The arousal system

# 2. DEFINE THE WIRING (Connections)
print("Wiring the network...")
connections = [
    # Thalamocortical Loops (The main highways)
    Connection(thalamus, frontal, weight=15.0),
    Connection(frontal, thalamus, weight=10.0),
    Connection(thalamus, occipital, weight=15.0),
    Connection(occipital, thalamus, weight=10.0),
    
    # Cortico-Cortical (The thinking cross-talk)
    Connection(frontal, occipital, weight=5.0),
    Connection(occipital, frontal, weight=5.0),
    
    # Arousal (The wake-up signal)
    Connection(reticular, thalamus, weight=20.0)
]

# 3. CREATE THE COMPUTATIONAL GRAPH
graph = ComputationalGraph(
    populations=[frontal, occipital, thalamus, reticular],
    connections=connections,
    dt=0.001,
    seed=42,
)

# 4. RUN SIMULATION
print("Running simulation (1.0 second)...")
results = graph.simulate(seconds=1.0, as_dict=True)

# 5. VISUALIZATION
plt.figure(figsize=(12, 8))

for i, (name, data) in enumerate(results.items()):
    plt.subplot(4, 1, i+1)
    plt.plot(data, label=name)
    plt.ylabel("Potential (mV)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)

plt.xlabel("Time steps")
plt.suptitle("4-Region Brain Network Synchronization")
plt.tight_layout()
plt.savefig('network_results.png')
print("\nSimulation complete. Results saved to 'network_results.png'")
