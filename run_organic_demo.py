import numpy as np
import matplotlib.pyplot as plt
from src.graph import Population, Connection, ComputationalGraph

def simulate(noise_level):
    p = Population(A=3.25, B=22.0, sigma=noise_level)
    graph = ComputationalGraph([p], [], dt=0.001, seed=42)
    return graph.simulate(steps=1000)[:, 0]

# 1. GENERATE DATA
print("Simulating Robotic Brain (No noise)...")
smooth_data = simulate(noise_level=0)

print("Simulating Organic Brain (With noise)...")
organic_data = simulate(noise_level=150) # Cranking the noise for effect

# 2. PLOT
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(smooth_data, color='#3498db', linewidth=2)
plt.title("Robotic Brain (Perfect Math)")
plt.ylabel("Potential")
plt.grid(True, alpha=0.2)

plt.subplot(2, 1, 2)
plt.plot(organic_data, color='#2ecc71', linewidth=1)
plt.title("Organic Human-like Brain (Stochastic)")
plt.ylabel("Potential")
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('organic_comparison.png')
print("\nSimulation complete. Results saved to 'organic_comparison.png'")
