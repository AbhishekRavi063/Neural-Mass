from itertools import product
from pathlib import Path

import numpy as np

from dreams_kcomplex_validation import (
    load_excerpt,
    mask_to_events,
    score_events,
)
from src.event_detection import K_complex_detection


def total_score(excerpts, params):
    total_expert = 0
    total_detected = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for signal, sfreq, expert_events in excerpts:
        mask = K_complex_detection(signal, sampling_frequency=int(sfreq), **params)
        detected_events = mask_to_events(mask, sfreq)
        score = score_events(expert_events, detected_events)
        total_expert += score["expert"]
        total_detected += score["detected"]
        total_tp += score["tp"]
        total_fp += score["fp"]
        total_fn += score["fn"]

    precision = total_tp / total_detected if total_detected else 0.0
    recall = total_tp / total_expert if total_expert else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expert": total_expert,
        "detected": total_detected,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    folder = Path("data/dreams/DatabaseKcomplexes")
    excerpts = [load_excerpt(folder, idx) for idx in range(1, 11)]

    grid = {
        "threshold_std": [2.3, 2.5, 2.7, 2.9],
        "min_duration": [0.16, 0.20],
        "merge_gap": [0.30, 0.40],
        "event_padding": [0.12, 0.18],
        "min_event_duration": [0.35, 0.45],
        "max_event_duration": [1.6, 2.0],
        "min_peak_to_peak": [35.0, 45.0, 55.0],
    }

    keys = list(grid)
    best = None
    top = []
    total = np.prod([len(grid[key]) for key in keys])
    print(f"Testing {total} parameter combinations...")

    for values in product(*(grid[key] for key in keys)):
        params = dict(zip(keys, values))
        params["require_biphasic"] = True
        score = total_score(excerpts, params)
        row = (score["f1"], score, params)
        top.append(row)
        if best is None or row[0] > best[0]:
            best = row

    top = sorted(top, reverse=True, key=lambda row: row[0])[:10]
    print("\nTop 10 settings:")
    for rank, (f1, score, params) in enumerate(top, 1):
        print(
            f"{rank}. f1={f1:.3f} precision={score['precision']:.3f} "
            f"recall={score['recall']:.3f} detected={score['detected']} "
            f"tp={score['tp']} fp={score['fp']} fn={score['fn']} params={params}"
        )


if __name__ == "__main__":
    main()
