import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import get_sleep_stage_data, load_sample_sleep_data
from src.event_detection import K_complex_detection
from src.metrics import get_performance_report


"""
Optional legacy PhysioNet exploration.

This script is not part of the main project validation path. The current
K-complex benchmark uses the DREAMS dataset because it includes expert
K-complex event labels.
"""


PRESETS = {
    "conservative": {
        "threshold_std": 2.5,
        "min_duration": 0.20,
        "merge_gap": 0.35,
        "event_padding": 0.18,
    },
    "balanced": {
        "threshold_std": 2.0,
        "min_duration": 0.15,
        "merge_gap": 0.40,
        "event_padding": 0.20,
    },
    "sensitive": {
        "threshold_std": 1.6,
        "min_duration": 0.12,
        "merge_gap": 0.45,
        "event_padding": 0.22,
    },
}


def count_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return 0
    starts = mask & np.concatenate(([True], ~mask[:-1]))
    return int(np.sum(starts))


def event_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    return list(zip(starts, ends))


def detect_epoch(signal, sfreq, preset):
    return K_complex_detection(
        signal,
        sampling_frequency=sfreq,
        **PRESETS[preset],
    )


def scan_epochs(epochs, sfreq):
    summary = {}
    masks_by_preset = {}
    for preset in PRESETS:
        masks = [detect_epoch(signal, sfreq, preset) for signal in epochs]
        counts = [count_segments(mask) for mask in masks]
        summary[preset] = {
            "counts": counts,
            "epochs_with_events": sum(count > 0 for count in counts),
            "total_events": sum(counts),
            "best_epoch": int(np.argmax(counts)),
            "best_count": int(np.max(counts)),
        }
        masks_by_preset[preset] = masks
    return summary, masks_by_preset


def plot_epoch(signal, masks_by_preset, sfreq, epoch_index, filename):
    times = np.arange(len(signal)) / sfreq
    signal_uv = signal * 1e6
    colors = {
        "conservative": "#ffb3d9",
        "balanced": "#b8e986",
        "sensitive": "#9bdbff",
    }

    plt.figure(figsize=(13, 5))
    plt.plot(times, signal_uv, color="#2c3e50", linewidth=1.2, label="Real EEG")
    y_min, y_max = np.min(signal_uv), np.max(signal_uv)
    for preset, mask in masks_by_preset.items():
        plt.fill_between(
            times,
            y_min,
            y_max,
            where=mask,
            color=colors[preset],
            alpha=0.22,
            label=f"{preset.title()} detection",
        )
    plt.title(f"Real PhysioNet EEG - Stage 2 Epoch {epoch_index}")
    plt.xlabel("Time (s)")
    plt.ylabel("Potential (uV)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename)


def main():
    print("NOTE: This is an optional legacy PhysioNet exploration.")
    print("Main K-complex validation is DREAMS: run validate_dreams_pipeline.py.")
    print()

    raw = load_sample_sleep_data(subject=0, recording=[1])
    epochs = get_sleep_stage_data(raw, stage="Sleep stage 2")
    if epochs is None or len(epochs) == 0:
        raise RuntimeError("No Stage 2 epochs found.")

    sfreq = int(raw.info["sfreq"])
    summary, masks_by_preset = scan_epochs(epochs, sfreq)

    print("==========================================")
    print(" REAL EEG K-COMPLEX DETECTION ANALYSIS")
    print("==========================================")
    print(f"Stage-2 epochs: {len(epochs)}")
    print(f"Sampling frequency: {sfreq} Hz")
    print()

    for preset, result in summary.items():
        print(f"{preset.title()} preset")
        print(f"- epochs with events: {result['epochs_with_events']} / {len(epochs)}")
        print(f"- total events: {result['total_events']}")
        print(f"- best epoch: {result['best_epoch']} ({result['best_count']} events)")
        print(f"- counts: {result['counts']}")
        print()

    epoch_index = summary["balanced"]["best_epoch"]
    masks_for_plot = {
        preset: masks_by_preset[preset][epoch_index]
        for preset in PRESETS
    }
    report = get_performance_report(epochs[epoch_index], sampling_frequency=sfreq)
    print(f"Balanced best epoch metrics: {report}")
    print(f"Balanced best epoch segments: {event_segments(masks_for_plot['balanced'])}")

    plot_epoch(
        epochs[epoch_index],
        masks_for_plot,
        sfreq,
        epoch_index,
        "real_kcomplex_preset_comparison.png",
    )
    print("\nSaved real_kcomplex_preset_comparison.png")


if __name__ == "__main__":
    main()
