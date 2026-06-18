"""Demo script demonstrating traveling waves and closed-loop sleep pressure dynamics.

Run:
    python run_spatiotemporal_demo.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from neural_mass.models.spatiotemporal_model import SpatiotemporalThalamocorticalModel
from neural_mass.models.thalamocortical_model import ThalamocorticalParameters

# Define paths
ARTIFACT_DIR = Path(r"C:\Users\abhis\.gemini\antigravity\brain\0bcae6b5-01b4-4c39-acf8-3256438eabb9")
PLOT_PATH_WAVES = ARTIFACT_DIR / "spatiotemporal_waves.png"
PLOT_PATH_S = ARTIFACT_DIR / "closed_loop_transitions.png"

print("=" * 64)
print("RUNNING SPATIOTEMPORAL MODEL DEMONSTRATION")
print("=" * 64)

# ══════════════════════════════════════════════════════════════════════════════
# Part 1. Traveling Waves (Anterior-to-Posterior)
# ══════════════════════════════════════════════════════════════════════════════
print("\n1. Simulating traveling waves on 8-node sagittal chain...")
model = SpatiotemporalThalamocorticalModel(
    n_nodes=8,
    lateral_coupling_strength=1.6,
    spatial_spread=1.0,
    pacemaker_strength=0.35,
    seed=123,
)

# Simulate 8.0 seconds at 200 Hz
fs = 200
seconds = 8.0
out = model.simulate(seconds=seconds, sampling_frequency=fs, closed_loop=False)

cortical = out["cortical_pyramidal"]  # shape (steps, N)
t = np.arange(len(cortical)) / fs

# Create plots for traveling waves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Heatmap showing node cortical pyramidal activity
im = ax1.imshow(
    cortical.T,
    aspect="auto",
    cmap="RdBu_r",
    extent=[0, seconds, 8, 1],
    vmin=-0.4,
    vmax=0.2,
)
ax1.set_ylabel("Sagittal Grid Node\n(Anterior 1 -> Posterior 8)")
ax1.set_title("Spatiotemporal Cortical Pyramidal Propagation (Traveling Slow Waves)")
fig.colorbar(im, ax=ax1, label="Pyramidal State (AU)")

# Line plot of Fz, Cz, and Pz scalp projections
ax2.plot(t, out["eeg_fz"], label="Fz (Frontal)", color="royalblue", alpha=0.9)
ax2.plot(t, out["eeg_cz"], label="Cz (Central)", color="forestgreen", alpha=0.9)
ax2.plot(t, out["eeg_pz"], label="Pz (Parietal)", color="crimson", alpha=0.9)
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("EEG Scalp Projection (AU)")
ax2.set_title("Projected Scalp EEG Channels (Phase Lag: Fz leads Pz)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(PLOT_PATH_WAVES, dpi=150)
plt.close()
print(f"  Saved traveling wave plots to: {PLOT_PATH_WAVES}")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2. Closed-Loop Sleep Stage Transitions (Process S)
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. Simulating closed-loop Process S homeostatic sleep stage cycles...")
model_cl = SpatiotemporalThalamocorticalModel(
    n_nodes=8,
    lateral_coupling_strength=1.4,
    spatial_spread=1.0,
    pacemaker_strength=0.30,
    seed=42,
)

# Simulate 50.0 seconds at 200 Hz with fast accumulation/dissipation constants
seconds_cl = 50.0
out_cl = model_cl.simulate(
    seconds=seconds_cl,
    sampling_frequency=fs,
    closed_loop=True,
    tau_accum=10.0,
    tau_dissip=12.0,
    initial_sleep_pressure=0.85,
)

t_cl = np.arange(len(out_cl["eeg_cz"])) / fs
sleep_pressure = out_cl["sleep_pressure"]

# Calculate slow wave activity (SWA) envelope globally from velocity for visualization
vel = np.diff(out_cl["cortical_pyramidal"], axis=0) * fs
avg_abs_vel = np.zeros(len(out_cl["eeg_cz"]))
# Match length
avg_abs_vel[1:] = np.mean(np.abs(vel), axis=1)
# smooth slightly for plotting
envelope = np.zeros_like(avg_abs_vel)
val = 0.0
for i in range(len(avg_abs_vel)):
    val += (avg_abs_vel[i] - val) / (2.0 * fs)
    envelope[i] = val

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Cz EEG signal (which changes amplitude as S cycles)
ax1.plot(t_cl, out_cl["eeg_cz"], color="purple", alpha=0.8)
ax1.set_ylabel("Cz EEG Signal (AU)")
ax1.set_title("EEG Activity over Sleep Cycles (Delta Waves decay as Sleep Pressure drops)")
ax1.grid(True, linestyle="--", alpha=0.5)

# Process S sleep pressure and SWA envelope
ax2.plot(t_cl, sleep_pressure, label="Process S (Sleep Pressure)", color="darkorange", linewidth=2.5)
ax2.plot(t_cl, envelope * 2.0, label="Slow Wave Activity (SWA) x 2", color="teal", linestyle="--", alpha=0.9)
ax2.axhline(0.10, color="gray", linestyle=":", label="Arousal Threshold")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("State Level")
ax2.set_title("Process S Homeostatic Sleep-Wake Transitions (Self-Organizing Stage Cycling)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(PLOT_PATH_S, dpi=150)
plt.close()
print(f"  Saved sleep cycling plots to: {PLOT_PATH_S}")

print("\n" + "=" * 64)
print("DEMONSTRATION RUN COMPLETE")
print("=" * 64)
