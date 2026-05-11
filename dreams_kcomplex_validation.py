import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.event_detection import K_complex_detection
from src.event_scoring import aggregate_scores, event_iou, score_events


PRESETS = {
    "conservative": {
        "threshold_std": 2.5,
        "min_duration": 0.20,
        "merge_gap": 0.35,
        "event_padding": 0.18,
    },
    "strict": {
        "threshold_std": 2.7,
        "min_duration": 0.18,
        "merge_gap": 0.35,
        "event_padding": 0.18,
        "min_peak_to_peak": 45.0,
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


def read_scoring_file(path):
    events = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        onset = float(parts[0])
        duration = float(parts[1])
        events.append({"onset": onset, "end": onset + duration, "duration": duration})
    return events


def mask_to_events(mask, sfreq):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    return [
        {"onset": start / sfreq, "end": (end + 1) / sfreq, "duration": (end - start + 1) / sfreq}
        for start, end in zip(starts, ends)
    ]


def read_signal_txt(path):
    values = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        values.append(float(line))
    return np.asarray(values)


def load_excerpt(folder, excerpt_number):
    signal_path = folder / f"excerpt{excerpt_number}.txt"
    scoring_path = folder / f"Visual_scoring1_excerpt{excerpt_number}.txt"
    signal = read_signal_txt(signal_path)
    sfreq = 200.0
    expert_events = read_scoring_file(scoring_path)
    return signal, sfreq, expert_events


def evaluate_excerpt(folder, excerpt_number, preset):
    signal, sfreq, expert_events = load_excerpt(folder, excerpt_number)
    mask = K_complex_detection(signal, sampling_frequency=int(sfreq), **PRESETS[preset])
    detected_events = mask_to_events(mask, sfreq)
    scores = score_events(expert_events, detected_events)
    return signal, sfreq, expert_events, detected_events, mask, scores


def plot_excerpt(signal, sfreq, expert_events, detected_mask, excerpt_number, preset, output):
    times = np.arange(len(signal)) / sfreq
    expert_mask = np.zeros(len(signal), dtype=bool)
    for event in expert_events:
        start = max(0, int(event["onset"] * sfreq))
        end = min(len(signal), int(event["end"] * sfreq))
        expert_mask[start:end] = True

    if expert_events:
        window_start = max(0.0, expert_events[0]["onset"] - 10.0)
        window_end = min(times[-1], window_start + 40.0)
    else:
        window_start = 0.0
        window_end = min(times[-1], 40.0)
    window_mask = (times >= window_start) & (times <= window_end)

    plt.figure(figsize=(13, 5))
    plt.plot(times[window_mask], signal[window_mask], color="#2c3e50", linewidth=1.0, label="DREAMS EEG CZ-A1")
    y_min, y_max = np.percentile(signal[window_mask], [1, 99])
    plt.fill_between(
        times[window_mask],
        y_min,
        y_max,
        where=expert_mask[window_mask],
        color="#9bdbff",
        alpha=0.32,
        label="Expert K-complex",
    )
    plt.fill_between(
        times[window_mask],
        y_min,
        y_max,
        where=detected_mask[window_mask],
        color="#ffb3d9",
        alpha=0.32,
        label="Detected K-complex",
    )
    plt.ylim(y_min, y_max)
    plt.title(f"DREAMS K-complex Validation - Excerpt {excerpt_number} ({preset}, zoom)")
    plt.xlabel("Time (s)")
    plt.ylabel("Potential (uV)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)


def main():
    parser = argparse.ArgumentParser(description="Validate K-complex detector on DREAMS.")
    parser.add_argument("--folder", default="data/dreams/DatabaseKcomplexes")
    parser.add_argument("--preset", choices=PRESETS, default="balanced")
    parser.add_argument("--excerpt", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Evaluate all 10 excerpts.")
    args = parser.parse_args()

    folder = Path(args.folder)
    excerpt_numbers = range(1, 11) if args.all else [args.excerpt]
    all_scores = []

    for excerpt_number in excerpt_numbers:
        signal, sfreq, expert_events, detected_events, mask, scores = evaluate_excerpt(
            folder,
            excerpt_number,
            args.preset,
        )
        all_scores.append(scores)
        print(
            f"excerpt={excerpt_number} "
            f"expert={scores['expert']} detected={scores['detected']} "
            f"tp={scores['tp']} fp={scores['fp']} fn={scores['fn']} "
            f"precision={scores['precision']:.3f} recall={scores['recall']:.3f} "
            f"f1={scores['f1']:.3f} mean_iou={scores['mean_iou']:.3f}"
        )

        if excerpt_number == args.excerpt:
            plot_excerpt(
                signal,
                sfreq,
                expert_events,
                mask,
                excerpt_number,
                args.preset,
                "dreams_kcomplex_validation.png",
            )

    if args.all:
        totals = aggregate_scores(all_scores)
        print("\nTOTAL")
        print(f"expert={totals['expert']} detected={totals['detected']} tp={totals['tp']} fp={totals['fp']} fn={totals['fn']}")
        print(f"precision={totals['precision']:.3f} recall={totals['recall']:.3f} f1={totals['f1']:.3f}")

    print("\nSaved dreams_kcomplex_validation.png")


if __name__ == "__main__":
    main()
