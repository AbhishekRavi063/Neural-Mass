import numpy as np
import matplotlib.pyplot as plt
from src.inference import find_best_parameters, run_simulation
from src.metrics import get_performance_report

print("==========================================")
print("   NEURAL MASS MODEL: MASTER DASHBOARD   ")
print("==========================================\n")

# 1. PREPARE THE TARGET DATA
print("[1/4] Generating synthetic target data (no noise)...")
# Target has secret parameters: A=5.0, B=35.0
target_signal = run_simulation(A=5.0, B=35.0, steps=1000, seed=7)
target_report = get_performance_report(target_signal)

# 2. RUN AUTOMATED DISCOVERY (Optuna)
print("[2/4] Running High-Precision Discovery (100 Trials)...")
best_params, best_error = find_best_parameters(target_signal, n_trials=100, seed=7)

# 3. GENERATE THE BEST-FIT SIMULATION
print("[3/4] Generating Best-Fit Robotic Simulation...")
best_sim = run_simulation(A=best_params['A'], B=best_params['B'], steps=1000, seed=7)
final_report = get_performance_report(best_sim, target=target_signal)

# 4. CREATE THE PROFESSIONAL REPORT
print("[4/4] Creating Final Research Report...")
plt.style.use('dark_background') # Premium look
fig = plt.figure(figsize=(15, 10))
plt.suptitle("BRAIN STATE DISCOVERY REPORT", fontsize=20, fontweight='bold', color='#3498db')

# Panel 1: The Master Comparison (Waves)
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax1.plot(target_signal, label="SYNTHETIC TARGET", color='#95a5a6', alpha=0.5, linewidth=3)
ax1.plot(best_sim, label="OPTIMIZED SIMULATOR", color='#3498db', linewidth=1.5)
ax1.set_title("Waveform Alignment (Similarity: {}%)".format(int(final_report['Similarity']*100)))
ax1.legend()
ax1.grid(True, alpha=0.1)

# Panel 2: Discovery Metrics (Text Scorecard)
ax2 = plt.subplot2grid((3, 3), (1, 0))
ax2.axis('off')
ax2.text(0, 0.9, "DISCOVERED BIOLOGY", fontsize=14, fontweight='bold', color='#f1c40f')
ax2.text(0, 0.7, f"Parameter A: {round(best_params['A'], 2)}", fontsize=12)
ax2.text(0, 0.6, f"Parameter B: {round(best_params['B'], 2)}", fontsize=12)
ax2.text(0, 0.4, "QUALITY SCORES", fontsize=14, fontweight='bold', color='#2ecc71')
ax2.text(0, 0.2, f"Rhythmicity: {final_report['Rhythmicity (0-1)']}", fontsize=12)
ax2.text(0, 0.1, f"RMSE Error: {final_report['RMSE']}", fontsize=12)

# Panel 3: Correlation Plot
ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=2)
ax3.scatter(target_signal, best_sim, s=2, alpha=0.3, color='#e74c3c')
ax3.set_title("Statistical Correlation (Fit Quality)")
ax3.set_xlabel("Target Potential")
ax3.set_ylabel("Simulator Potential")

# Panel 4: FFT / Frequency Spectrum (Agnostic Insight)
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
fft_target = np.abs(np.fft.fft(target_signal))[:100]
fft_sim = np.abs(np.fft.fft(best_sim))[:100]
ax4.fill_between(range(len(fft_target)), fft_target, color='#95a5a6', alpha=0.2, label="Target Spectrum")
ax4.plot(fft_sim, color='#3498db', label="Simulator Spectrum")
ax4.set_title("Frequency Power Analysis (Alpha/Theta Sync)")
ax4.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Final_Brain_Report.png')

print("\n==========================================")
print("   DASHBOARD COMPLETE: REPORT GENERATED   ")
print("   Check 'Final_Brain_Report.png'         ")
print("==========================================")
