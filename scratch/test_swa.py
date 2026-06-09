import numpy as np
from neural_mass.models.spatiotemporal_model import SpatiotemporalThalamocorticalModel

model = SpatiotemporalThalamocorticalModel(n_nodes=8, seed=42)
out = model.simulate(
    seconds=15.0,
    sampling_frequency=200,
    closed_loop=True,
    tau_accum=8.0,
    tau_dissip=10.0,
    initial_sleep_pressure=0.85
)

sleep_pressure = out["sleep_pressure"]
# We also want to see the envelope values, but since they are not returned directly, 
# let's calculate the cortical velocity from the downsampled output and estimate SWA.
cortical = out["cortical_pyramidal"]
velocity = np.diff(cortical, axis=0) * 200.0
avg_abs_vel = np.mean(np.abs(velocity), axis=1)

print("Average Absolute Velocity over time (deciles):")
for q in range(0, 101, 10):
    print(f"  {q}%: {np.percentile(avg_abs_vel, q):.6f}")

print(f"Initial sleep pressure: {sleep_pressure[0]:.4f}")
print(f"Final sleep pressure: {sleep_pressure[-1]:.4f}")
print(f"Max sleep pressure: {sleep_pressure.max():.4f}")
