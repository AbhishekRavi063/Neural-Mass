import numpy as np
import matplotlib.pyplot as plt

from src.inference import find_best_parameters, run_simulation
from src.metrics import get_performance_report


def dominant_frequency(signal, sampling_frequency):
    centered = signal - np.mean(signal)
    frequencies = np.fft.rfftfreq(len(centered), d=1 / sampling_frequency)
    power = np.abs(np.fft.rfft(centered))
    if len(power) <= 1:
        return 0.0
    idx = np.argmax(power[1:]) + 1
    return frequencies[idx]


def main():
    sampling_frequency = 1000
    steps = 1000

    target = run_simulation(A=4.0, B=40.0, steps=steps, seed=11)
    best_params, best_error = find_best_parameters(target, n_trials=100, seed=11)
    fitted = run_simulation(
        A=best_params["A"],
        B=best_params["B"],
        steps=steps,
        seed=11,
    )

    report = get_performance_report(
        fitted,
        target=target,
        sampling_frequency=sampling_frequency,
    )
    frequency = dominant_frequency(fitted, sampling_frequency)

    checks = {
        "finite_signal": bool(np.isfinite(fitted).all()),
        "non_flat_signal": bool(np.std(fitted) > 1.0),
        "high_similarity": bool(report["Similarity"] >= 0.95),
        "low_rmse": bool(report["RMSE"] <= 2.0),
        "physiological_demo_frequency": bool(1.0 <= frequency <= 40.0),
    }

    print("==========================================")
    print(" FIRST-LAYER VALIDATION")
    print("==========================================")
    print(f"Best parameters: A={best_params['A']:.3f}, B={best_params['B']:.3f}")
    print(f"Best RMSE from optimizer: {best_error:.4f}")
    print(f"Performance report: {report}")
    print(f"Dominant frequency: {frequency:.2f} Hz")
    print("\nChecks:")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"- {name}: {status}")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(target, label="Target", color="#7f8c8d", linewidth=2, alpha=0.8)
    plt.plot(fitted, label="Fitted", color="#2980b9", linewidth=1.5)
    plt.title("First-Layer Validation: Target vs Fitted Signal")
    plt.ylabel("Potential")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.subplot(2, 1, 2)
    target_power = np.abs(np.fft.rfft(target - np.mean(target)))
    fitted_power = np.abs(np.fft.rfft(fitted - np.mean(fitted)))
    frequencies = np.fft.rfftfreq(len(fitted), d=1 / sampling_frequency)
    plt.plot(frequencies[:80], target_power[:80], label="Target Spectrum", color="#7f8c8d")
    plt.plot(frequencies[:80], fitted_power[:80], label="Fitted Spectrum", color="#2980b9")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("first_layer_validation.png")
    print("\nSaved first_layer_validation.png")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
