import numpy as np
from src.graph import Population, Connection, ComputationalGraph
from src.metrics import get_performance_report

def simulate_brain(noise_level, connection_weight):
    cortex = Population()
    thalamus = Population()
    conns = [Connection(cortex, thalamus, weight=connection_weight), 
             Connection(thalamus, cortex, weight=connection_weight)]
    graph = ComputationalGraph(
        [cortex, thalamus],
        conns,
        dt=0.001,
        input_std=noise_level,
        seed=42,
    )
    return graph.simulate(steps=1000)[:, 0]

# 1. GENERATE GOOD DATA (Steady and Strong)
print("--- TESTING GOOD BRAIN ---")
good_brain = simulate_brain(noise_level=5, connection_weight=10.0)
good_report = get_performance_report(good_brain)
print(f"Metrics: {good_report}")

# 2. GENERATE BAD DATA (Messy and Weak)
print("\n--- TESTING BAD BRAIN ---")
bad_brain = simulate_brain(noise_level=100, connection_weight=1.0)
bad_report = get_performance_report(bad_brain)
print(f"Metrics: {bad_report}")

# CONCLUSION
if good_report["Rhythmicity (0-1)"] > bad_report["Rhythmicity (0-1)"]:
    print("\nSUCCESS: The Metrics correctly identified the Good Brain!")
else:
    print("\nWARNING: Metrics need more tuning.")
