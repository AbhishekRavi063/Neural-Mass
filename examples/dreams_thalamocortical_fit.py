from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dreams_kcomplex_validation import load_excerpt
from src.thalamocortical_fitting import (
    FEATURE_NAMES,
    build_condition_summary,
    fit_thalamocortical_features,
    parameter_difference,
)


DREAMS_FOLDER = Path("data/dreams/DatabaseKcomplexes")
WINDOW_SECONDS = 5.0
MAX_WINDOWS = 60


def event_overlaps(start, end, events):
    for event in events:
        if max(start, event["onset"]) < min(end, event["end"]):
            return True
    return False


def extract_event_windows(signal, sfreq, events, window_seconds=WINDOW_SECONDS):
    half = window_seconds / 2
    windows = []
    for event in events:
        center = (event["onset"] + event["end"]) / 2
        start_time = center - half
        end_time = center + half
        if start_time < 0 or end_time > len(signal) / sfreq:
            continue
        start = int(round(start_time * sfreq))
        end = start + int(round(window_seconds * sfreq))
        windows.append(signal[start:end])
    return windows


def extract_matched_control_windows(signal, sfreq, events, count, window_seconds=WINDOW_SECONDS):
    duration = len(signal) / sfreq
    step = window_seconds
    windows = []
    start_time = 0.0
    while start_time + window_seconds <= duration and len(windows) < count:
        end_time = start_time + window_seconds
        if not event_overlaps(start_time, end_time, events):
            start = int(round(start_time * sfreq))
            end = start + int(round(window_seconds * sfreq))
            windows.append(signal[start:end])
        start_time += step
    return windows


def collect_dreams_windows(folder=DREAMS_FOLDER, max_windows=MAX_WINDOWS):
    kcomplex_windows = []
    control_windows = []
    sfreq = 200.0

    for excerpt_number in range(1, 11):
        signal, sfreq, expert_events = load_excerpt(folder, excerpt_number)
        excerpt_kc = extract_event_windows(signal, sfreq, expert_events)
        remaining = max(0, max_windows - len(kcomplex_windows))
        kcomplex_windows.extend(excerpt_kc[:remaining])

        excerpt_controls = extract_matched_control_windows(
            signal,
            sfreq,
            expert_events,
            count=len(excerpt_kc),
        )
        remaining_controls = max(0, max_windows - len(control_windows))
        control_windows.extend(excerpt_controls[:remaining_controls])

        if len(kcomplex_windows) >= max_windows and len(control_windows) >= max_windows:
            break

    return kcomplex_windows[:max_windows], control_windows[:max_windows], sfreq


def print_features(title, features):
    print(title)
    for name in FEATURE_NAMES:
        print(f"- {name}: {features[name]:.6g}")


def plot_comparison(kc_features, control_features, kc_params, control_params, output):
    feature_values = [kc_features[name] for name in FEATURE_NAMES]
    control_values = [control_features[name] for name in FEATURE_NAMES]
    feature_ratio = np.asarray(feature_values) / (np.asarray(control_values) + 1e-8)

    parameter_names = [
        "adaptation_strength",
        "cortex_to_thalamus",
        "thalamus_to_cortex",
        "reticular_inhibition",
        "background_drive",
        "cortical_damping",
        "spindle_damping",
    ]
    kc_param_values = np.asarray([getattr(kc_params, name) for name in parameter_names])
    control_param_values = np.asarray([getattr(control_params, name) for name in parameter_names])

    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(np.arange(len(FEATURE_NAMES)), feature_ratio, color="#2a9d8f")
    ax1.axhline(1.0, color="#264653", linestyle="--", linewidth=1.0)
    ax1.set_xticks(np.arange(len(FEATURE_NAMES)))
    ax1.set_xticklabels(FEATURE_NAMES, rotation=40, ha="right")
    ax1.set_title("DREAMS K-complex / Control Feature Ratio")
    ax1.set_ylabel("Ratio")
    ax1.grid(True, axis="y", alpha=0.2)

    ax2 = plt.subplot(1, 2, 2)
    width = 0.38
    x = np.arange(len(parameter_names))
    ax2.bar(x - width / 2, kc_param_values, width, label="K-complex fit", color="#e76f51")
    ax2.bar(x + width / 2, control_param_values, width, label="Control fit", color="#457b9d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(parameter_names, rotation=40, ha="right")
    ax2.set_title("Fitted Compact Thalamocortical Parameters")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output)


def main():
    print("==========================================")
    print(" DREAMS THALAMOCORTICAL WINDOW FIT")
    print("==========================================")
    kcomplex_windows, control_windows, sfreq = collect_dreams_windows()
    if not kcomplex_windows or not control_windows:
        raise RuntimeError("Could not collect both K-complex and control windows.")

    print(f"K-complex windows: {len(kcomplex_windows)}")
    print(f"Control windows: {len(control_windows)}")
    print(f"Window length: {WINDOW_SECONDS:.1f} s")
    print(f"Sampling frequency: {sfreq:.0f} Hz")
    print()

    kc_summary = build_condition_summary(kcomplex_windows, sfreq)
    control_summary = build_condition_summary(control_windows, sfreq)
    print_features("K-complex feature summary", kc_summary)
    print()
    print_features("Control feature summary", control_summary)

    print("\nFitting compact thalamocortical model to K-complex feature summary...")
    kc_params, kc_fit_features, kc_error = fit_thalamocortical_features(
        kc_summary,
        seconds=WINDOW_SECONDS,
        sfreq=int(sfreq),
        n_trials=60,
        seed=31,
    )
    print("Fitting compact thalamocortical model to control feature summary...")
    control_params, control_fit_features, control_error = fit_thalamocortical_features(
        control_summary,
        seconds=WINDOW_SECONDS,
        sfreq=int(sfreq),
        n_trials=60,
        seed=47,
    )

    print("\nFitted K-complex parameters")
    for name, value in kc_params.__dict__.items():
        if name != "dt":
            print(f"- {name}: {value:.6g}")
    print(f"- feature_fit_error: {kc_error:.6g}")

    print("\nFitted control parameters")
    for name, value in control_params.__dict__.items():
        if name != "dt":
            print(f"- {name}: {value:.6g}")
    print(f"- feature_fit_error: {control_error:.6g}")

    print("\nK-complex minus control parameter difference")
    for name, value in parameter_difference(kc_params, control_params).items():
        print(f"- {name}: {value:+.6g}")

    plot_comparison(
        kc_summary,
        control_summary,
        kc_params,
        control_params,
        "dreams_thalamocortical_fit.png",
    )
    print("\nSaved dreams_thalamocortical_fit.png")


if __name__ == "__main__":
    main()
