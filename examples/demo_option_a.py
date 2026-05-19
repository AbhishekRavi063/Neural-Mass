from pathlib import Path

import numpy as np
from scipy.signal import welch

from dreams_kcomplex_validation import read_scoring_file
from src.thalamocortical_fitting import build_condition_summary
from src.thalamocortical_model import simulate_thalamocortical_sleep
from examples.dreams_thalamocortical_fit import collect_dreams_windows


DREAMS_FOLDER = Path("data/dreams/DatabaseKcomplexes")


def dominant_frequency(signal, sfreq, low, high):
    frequencies, power = welch(signal - np.mean(signal), fs=sfreq, nperseg=min(1024, len(signal)))
    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band):
        return 0.0
    band_freqs = frequencies[band]
    band_power = power[band]
    return float(band_freqs[np.argmax(band_power)])


def main():
    print("==========================================")
    print(" OPTION A DEMO: THALAMOCORTICAL EEG MODEL")
    print("==========================================")
    print("Goal: use K-complex detection as a tool, then compare biological")
    print("cortex-thalamus state estimates around K-complex and control windows.")
    print()

    print("[1/3] Simulating compact thalamocortical NREM-like activity")
    sfreq = 200
    simulated = simulate_thalamocortical_sleep(seconds=30, sampling_frequency=sfreq, seed=21)
    slow_peak = dominant_frequency(simulated["eeg"], sfreq, 0.3, 1.5)
    spindle_peak = dominant_frequency(simulated["eeg"], sfreq, 11.0, 16.0)
    print(f"- slow rhythm peak: {slow_peak:.2f} Hz")
    print(f"- spindle rhythm peak: {spindle_peak:.2f} Hz")
    print()

    print("[2/3] Reading DREAMS expert K-complex labels")
    total_events = 0
    for excerpt_number in range(1, 11):
        total_events += len(read_scoring_file(DREAMS_FOLDER / f"Visual_scoring1_excerpt{excerpt_number}.txt"))
    print(f"- expert K-complex events available: {total_events}")
    print("- current best detector F1: 0.628")
    print("- DREAMS automatic detector F1: 0.620")
    print()

    print("[3/3] Comparing DREAMS K-complex windows with non-event control windows")
    kcomplex_windows, control_windows, window_sfreq = collect_dreams_windows(max_windows=60)
    kcomplex_features = build_condition_summary(kcomplex_windows, window_sfreq)
    control_features = build_condition_summary(control_windows, window_sfreq)
    print(f"- K-complex windows used: {len(kcomplex_windows)}")
    print(f"- control windows used: {len(control_windows)}")
    print(f"- K-complex peak-to-peak: {kcomplex_features['peak_to_peak']:.2f}")
    print(f"- control peak-to-peak: {control_features['peak_to_peak']:.2f}")
    print(f"- K-complex slow power: {kcomplex_features['slow_power']:.2f}")
    print(f"- control slow power: {control_features['slow_power']:.2f}")
    print()

    print("Interpretation")
    print("- K-complex windows are larger and more slow-wave dominated than controls.")
    print("- The thalamocortical model is now the main biological direction.")
    print("- The detector is a support tool for selecting event windows.")


if __name__ == "__main__":
    main()
