import numpy as np
from src.inference import find_best_parameters, run_simulation
from src.metrics import get_performance_report

# 1. CREATE A TARGET SIGNAL (The "Patient")
# Let's say the patient has hidden parameters A=4.0 and B=40.0
print("--- GENERATING TARGET BRAIN (SECRET PARAMETERS) ---")
SECRET_A = 4.0
SECRET_B = 40.0
target_signal = run_simulation(A=SECRET_A, B=SECRET_B, steps=1000, seed=11)
print(f"Target Brain Generated (A is hidden, B is hidden)")

# 2. START THE OPTUNA DETECTIVE
print("\n--- STARTING AUTOMATED TUNING (OPTUNA) ---")
print("Trying to find the secret A and B values...")
best_params, best_error = find_best_parameters(target_signal, n_trials=100, seed=11)

# 3. SHOW THE RESULTS
print("\n--- RESULTS FOUND BY THE COMPUTER ---")
print(f"Secret A was {SECRET_A} | Optuna found: {round(best_params['A'], 2)}")
print(f"Secret B was {SECRET_B} | Optuna found: {round(best_params['B'], 2)}")
print(f"Final Error (RMSE): {best_error}")

# 4. FINAL QUALITY CHECK
final_sim = run_simulation(A=best_params['A'], B=best_params['B'], steps=1000, seed=11)
final_report = get_performance_report(final_sim, target=target_signal)
print(f"\nFinal Validation Report: {final_report}")

if final_report["Similarity"] > 0.9:
    print("\nSUCCESS: The Computer matched the brain state perfectly!")
else:
    print("\nWARNING: Tuning needs more trials.")
