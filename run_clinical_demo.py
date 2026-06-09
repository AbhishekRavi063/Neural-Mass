"""Demo script demonstrating clinical diagnostic features of the Neural-Mass library.

Run:
    python run_clinical_demo.py
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from neural_mass.utils.dreams_io import read_signal_txt
from neural_mass.utils.preprocessing import clinical_artifact_filter, adaptive_scale
from neural_mass.detection.clinical import compute_so_pac, estimate_thalamic_gating
from neural_mass.inference.thalamocortical_fitting import fit_thalamocortical_multi_objective, fit_schizophrenia
from neural_mass import ThalamocorticalSleepModel

# Define paths
ARTIFACT_DIR = Path(r"C:\Users\abhis\.gemini\antigravity\brain\0bcae6b5-01b4-4c39-acf8-3256438eabb9")
PLOT_PATH_PAC = ARTIFACT_DIR / "clinical_pac_coupling.png"
DREAMS_SIGNAL = Path("data/dreams/DatabaseKcomplexes/excerpt1.txt")

print("=" * 64)
print("RUNNING CLINICAL DIAGNOSTICS DEMONSTRATION")
print("=" * 64)

# ══════════════════════════════════════════════════════════════════════════════
# Part 1. Load and Clean Real EEG Data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n1. Loading and filtering real sleep EEG (DREAMS excerpt 1)...")
raw_signal = read_signal_txt(DREAMS_SIGNAL)
sfreq = 200.0

# Preprocess with clinical artifact filter (0.1 - 45 Hz bandpass + epoch variance rejection)
clean_signal = clinical_artifact_filter(raw_signal, sfreq=sfreq, reject_std_threshold=4.5)

print(f"  Signal length   : {len(raw_signal)/sfreq:.1f} seconds")
print(f"  Raw Signal std  : {np.std(raw_signal):.2f} uV")
print(f"  Clean Signal std: {np.std(clean_signal):.2f} uV")

# ══════════════════════════════════════════════════════════════════════════════
# Part 2. Phase-Amplitude Coupling (PAC) Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. Computing Slow-Oscillation Spindle Phase Coupling (SO-PAC)...")
# Extract a 5-minute N2 segment rich in spindles/slow-waves (from minute 10 to 15)
start_sample = int(10 * 60 * sfreq)
end_sample = int(15 * 60 * sfreq)
n2_segment = clean_signal[start_sample:end_sample]

pac_results = compute_so_pac(n2_segment, sfreq=sfreq, n_bins=18)

print(f"  Modulation Index (MI) : {pac_results['modulation_index']:.6f}")
print(f"  Preferred Phase       : {pac_results['preferred_phase_deg']:.1f} degrees")

# Plot the phase-amplitude distribution
plt.figure(figsize=(8, 5))
# Map rad to deg for centers
deg_centers = np.degrees(pac_results["bin_centers"])
plt.bar(
    deg_centers,
    pac_results["bin_amplitudes"],
    width=360.0 / 18,
    color="teal",
    edgecolor="black",
    alpha=0.8,
)
plt.axvline(pac_results["preferred_phase_deg"], color="red", linestyle="--", linewidth=2, label="Preferred Phase")
plt.xlabel("Slow Oscillation Phase (degrees)")
plt.ylabel("Mean Spindle Amplitude (uV)")
plt.title(f"SO-Spindle Phase-Amplitude Coupling (MI = {pac_results['modulation_index']:.6f})")
plt.xticks(np.arange(-180, 181, 45))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH_PAC, dpi=150)
plt.close()
print(f"  Saved PAC coupling histogram to: {PLOT_PATH_PAC}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 3. Disease-Specific Parameter Fitting Templates
# ══════════════════════════════════════════════════════════════════════════════
print("\n3. Running disease-specific template fitting (10 Optuna trials)...")
# Take a 30s epoch for fitting and normalize to arbitrary unit scale (~0.05 std)
fit_window_raw = clean_signal[start_sample : start_sample + int(30 * sfreq)]
fit_window_norm, scale_factor = adaptive_scale(fit_window_raw, target_std=0.05)

# Fit template 1: Healthy (standard multi-objective fit, 10 trials)
print("\n  Fitting healthy sleep model template...")
healthy_params, _, healthy_err = fit_thalamocortical_multi_objective(
    fit_window_norm, sfreq=int(sfreq), n_trials=10, seed=42
)

# Fit template 2: Schizophrenia template (TRN deficit + synaptic pruning constraints, 10 trials)
print("\n  Fitting schizophrenia template (TRN + synaptic pruning constraints)...")
sz_params, _, sz_err = fit_schizophrenia(
    fit_window_norm, sfreq=int(sfreq), n_trials=10, seed=42
)

# Compare parameter values
print("\n  Fitted parameters comparison:")
print(f"    {'Parameter':30s}  {'Healthy Fit':>12s}  {'Schizophrenia':>12s}")
print(f"    {'-'*58}")
print(f"    {'reticular_inhibition':30s}  {healthy_params.reticular_inhibition:12.4f}  {sz_params.reticular_inhibition:12.4f}  (Constrained [0.15, 0.45])")
print(f"    {'cortical_excitation_scale':30s}  {healthy_params.cortical_excitation_scale:12.4f}  {sz_params.cortical_excitation_scale:12.4f}  (Constrained [8.0, 15.0])")
print(f"    {'cortex_to_thalamus':30s}  {healthy_params.cortex_to_thalamus:12.4f}  {sz_params.cortex_to_thalamus:12.4f}")

print(f"\n  Template Fit Errors (L1 norm):")
print(f"    Healthy Sleep Template Fit Error       : {healthy_err:.4f}")
print(f"    Schizophrenia Pathological Fit Error  : {sz_err:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Part 4. Thalamocortical Gating Index (TGI) Estimation
# ══════════════════════════════════════════════════════════════════════════════
print("\n4. Reconstructing hidden subcortical states & estimating sensory gating...")
# Simulate using the fitted healthy model parameters and check gating
model_healthy = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
# Apply fitted parameters
model_healthy.parameters_ = healthy_params
sim_out = model_healthy.simulate(seconds=30.0, sampling_frequency=int(sfreq))

# Compute TGI
tgi = estimate_thalamic_gating(sim_out, reticular_threshold=0.65)
print(f"  Thalamocortical Gating Index (TGI)    : {tgi:.4f}  (fraction of sleep spindles gating sensory input)")

print("\n" + "=" * 64)
print("CLINICAL DIAGNOSTICS DEMONSTRATION RUN COMPLETE")
print("=" * 64)
