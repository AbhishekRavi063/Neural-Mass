import numpy as np
import matplotlib.pyplot as plt
from src.graph import Population, Connection, ComputationalGraph

def run_and_plot(case_name, A, B, noise_std, filename, color):
    cortex = Population(A=A, B=B)
    thalamus = Population(A=A, B=B)
    conns = [Connection(cortex, thalamus, weight=10.0), Connection(thalamus, cortex, weight=10.0)]
    graph = ComputationalGraph(
        [cortex, thalamus],
        conns,
        dt=0.001,
        input_std=noise_std,
        seed=42,
    )
    history = graph.simulate(steps=1000)[:, 0]
    
    plt.figure(figsize=(10, 4))
    plt.plot(history, color=color)
    plt.title(f"Sleep Type: {case_name}")
    plt.xlabel("Time steps")
    plt.ylabel("Potential (mV)")
    plt.grid(True, alpha=0.2)
    plt.savefig(filename)
    print(f"Generated {filename}")

# SLEEP VARIATION 1: Light Sleep (Low inhibition, low noise)
run_and_plot("Light Sleeper (Small K-Complex)", A=3.25, B=35.0, noise_std=5, filename="kcomp_light.png", color="#f39c12")

# SLEEP VARIATION 2: Deep Sleep (High inhibition, classic K-Complex)
run_and_plot("Deep Sleeper (Massive K-Complex)", A=3.25, B=55.0, noise_std=20, filename="kcomp_deep.png", color="#8e44ad")

# SLEEP VARIATION 3: Fragmented Sleep (Mixed noise and inhibition)
run_and_plot("Fragmented Sleep (Messy Waves)", A=3.5, B=45.0, noise_std=60, filename="kcomp_messy.png", color="#c0392b")
