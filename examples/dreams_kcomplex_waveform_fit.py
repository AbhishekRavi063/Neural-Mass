from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.dreams_thalamocortical_fit import WINDOW_SECONDS, collect_dreams_windows
from src.thalamocortical_fitting import fit_thalamocortical_waveform, standardize_waveform


def average_standardized_waveform(windows):
    standardized = np.asarray([standardize_waveform(window) for window in windows])
    return np.mean(standardized, axis=0)


def main():
    print("==========================================")
    print(" DREAMS K-COMPLEX WAVEFORM-LEVEL FIT")
    print("==========================================")
    kcomplex_windows, control_windows, sfreq = collect_dreams_windows(max_windows=40)
    target = average_standardized_waveform(kcomplex_windows)
    control = average_standardized_waveform(control_windows)

    params, fitted, error = fit_thalamocortical_waveform(target, sfreq=int(sfreq), n_trials=40, seed=71)
    times = np.arange(len(target)) / sfreq - WINDOW_SECONDS / 2

    print(f"K-complex windows used: {len(kcomplex_windows)}")
    print(f"Control windows used: {len(control_windows)}")
    print(f"Waveform fit error: {error:.6g}")
    print("Fitted waveform-level parameters")
    for name, value in params.__dict__.items():
        if name != "dt":
            print(f"- {name}: {value:.6g}")

    plt.figure(figsize=(12, 5))
    plt.plot(times, target, color="#1d3557", linewidth=2.0, label="Average DREAMS K-complex")
    plt.plot(times, control, color="#8d99ae", linewidth=1.2, alpha=0.9, label="Average control")
    plt.plot(times, fitted, color="#e76f51", linewidth=1.5, label="Fitted thalamocortical model")
    plt.axvline(0, color="#264653", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.title("Waveform-Level Compact Thalamocortical Fit")
    plt.xlabel("Time from event center (s)")
    plt.ylabel("Standardized amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("dreams_kcomplex_waveform_fit.png")
    print("\nSaved dreams_kcomplex_waveform_fit.png")


if __name__ == "__main__":
    main()
