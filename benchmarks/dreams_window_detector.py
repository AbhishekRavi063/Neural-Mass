import argparse
from pathlib import Path

import numpy as np

from neural_mass.dreams_io import read_scoring_file, read_signal_txt
from neural_mass.event_scoring import aggregate_scores, bootstrap_f1_ci, score_events, score_events_onset
from neural_mass.kcomplex_window_detector import (
    build_window_dataset,
    train_balanced_window_classifier,
    windows_to_events,
)


def build_excerpt(folder, excerpt_number):
    folder = Path(folder)
    signal = read_signal_txt(folder / f"excerpt{excerpt_number}.txt")
    expert_events = read_scoring_file(folder / f"Visual_scoring1_excerpt{excerpt_number}.txt")
    filtered, windows, X, y = build_window_dataset(signal, 200.0, expert_events)
    return {
        "excerpt": excerpt_number,
        "signal": signal,
        "filtered": filtered,
        "sfreq": 200.0,
        "expert_events": expert_events,
        "windows": windows,
        "X": X,
        "y": y,
    }


def evaluate_window_detector(
    folder="data/dreams/DatabaseKcomplexes",
    threshold=0.50,
    spindle_rejection=True,
    morphology_filter=False,
):
    datasets = [build_excerpt(folder, idx) for idx in range(1, 11)]
    rows = []
    scores_iou = []
    scores_onset = []

    for test_idx, test_set in enumerate(datasets):
        train_sets = [dataset for idx, dataset in enumerate(datasets) if idx != test_idx]
        X_train = np.vstack([dataset["X"] for dataset in train_sets])
        y_train = np.concatenate([dataset["y"] for dataset in train_sets])
        model = train_balanced_window_classifier(X_train, y_train, random_state=100 + test_idx)
        probabilities = model.predict_proba(test_set["X"])[:, 1]
        detected = windows_to_events(
            test_set["windows"],
            probabilities,
            test_set["sfreq"],
            threshold=threshold,
            n_samples=len(test_set["signal"]),
            signal=test_set["filtered"] if (spindle_rejection or morphology_filter) else None,
            spindle_rejection=spindle_rejection,
            morphology_filter=morphology_filter,
        )
        score = score_events(test_set["expert_events"], detected)
        onset_score = score_events_onset(test_set["expert_events"], detected, tolerance=0.5)
        scores_iou.append(score)
        scores_onset.append(onset_score)
        rows.append((test_set, probabilities, detected, score))
        print(
            f"window_detector excerpt={test_set['excerpt']} windows={len(test_set['windows'])} "
            f"expert={score['expert']} detected={score['detected']} tp={score['tp']} "
            f"fp={score['fp']} fn={score['fn']} precision={score['precision']:.3f} "
            f"recall={score['recall']:.3f} f1={score['f1']:.3f}"
        )

    totals = aggregate_scores(scores_iou)
    onset_totals = aggregate_scores(scores_onset)
    ci = bootstrap_f1_ci(scores_iou)

    print("\nTOTAL (IoU matching)")
    print(
        f"expert={totals['expert']} detected={totals['detected']} tp={totals['tp']} "
        f"fp={totals['fp']} fn={totals['fn']} precision={totals['precision']:.3f} "
        f"recall={totals['recall']:.3f} f1={totals['f1']:.3f}"
    )
    print(f"F1 95% CI (bootstrap): [{ci['lower']:.3f}, {ci['upper']:.3f}]  std={ci['std']:.3f}")

    print("\nTOTAL (onset ±0.5s matching)")
    print(
        f"expert={onset_totals['expert']} detected={onset_totals['detected']} "
        f"tp={onset_totals['tp']} fp={onset_totals['fp']} fn={onset_totals['fn']} "
        f"precision={onset_totals['precision']:.3f} recall={onset_totals['recall']:.3f} "
        f"f1={onset_totals['f1']:.3f}"
    )
    return rows, totals


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate wavelet/TEO sliding-window K-complex detector on DREAMS."
    )
    parser.add_argument("--folder", default="data/dreams/DatabaseKcomplexes")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--no-spindle-rejection", dest="spindle_rejection", action="store_false")
    parser.set_defaults(spindle_rejection=True)
    args = parser.parse_args()
    evaluate_window_detector(
        args.folder, threshold=args.threshold, spindle_rejection=args.spindle_rejection
    )


if __name__ == "__main__":
    main()
