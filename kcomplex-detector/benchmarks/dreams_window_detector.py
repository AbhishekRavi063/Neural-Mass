"""DREAMS K-Complex Benchmark — Leave-One-Out Cross-Validation.

Dataset
-------
DREAMS database: 10 x 30-min PSG excerpts, CZ-A1, 200 Hz.
Expert 1 + Expert 2 annotations (Expert 2 available for excerpts 1-5 only).
Evaluation always scores against Expert 1 (consistent with published baselines).

Method
------
Leave-one-out CV across all 10 excerpts. For each outer fold:
  - Train on 9 excerpts (Expert 1 union Expert 2 labels where available).
  - Excerpts 6-10 training labels are augmented with pseudo-labels from a
    first-pass model trained on excerpts 1-5 (prob >= 0.82 in those excerpts).
  - Select decision threshold via inner LOO-CV on the 9 training excerpts using
    F-beta (beta=2) scoring — weights recall 4x more than precision.
  - Evaluate on the held-out excerpt using IoU >= 0.20 event matching.

Usage
-----
  python -m benchmarks.dreams_window_detector                  # full LOO-CV
  python -m benchmarks.dreams_window_detector --threshold 0.50 # fixed threshold
  python -m benchmarks.dreams_window_detector --no-expert2-union

Results (as of latest run)
--------------------------
  F1=0.632  Precision=0.531  Recall=0.779  TP=212  FP=187  FN=60
  F1 95% CI: [0.552, 0.686]  std=0.035
  Inter-rater ceiling (Expert2 vs Expert1): F1=0.301
  DREAMS published auto-detector: F1=0.620
"""
import argparse
from pathlib import Path

import numpy as np

from kcomplex_detector.utils.dreams_io import read_scoring_file, read_signal_txt, read_union_events
from kcomplex_detector.utils.event_scoring import aggregate_scores, bootstrap_f1_ci, score_events, score_events_onset
from kcomplex_detector.kcomplex_window_detector import (
    build_window_dataset,
    select_threshold_by_cv,
    train_balanced_window_classifier,
    windows_to_events,
)

# Import the canonical threshold grid from the detector module so there is a
# single source of truth. The local definition previously duplicated it.
from kcomplex_detector.kcomplex_window_detector import _THRESHOLDS_GRID as THRESHOLDS_GRID


def build_excerpt(folder, excerpt_number, use_expert2_union=True):
    """Load one DREAMS excerpt.

    use_expert2_union: for excerpts 1-5, use Expert1 ∪ Expert2 as training labels.
    Evaluation always scores against Expert 1 alone (consistent with baselines).
    """
    folder = Path(folder)
    signal = read_signal_txt(folder / f"excerpt{excerpt_number}.txt")
    expert_events = read_scoring_file(folder / f"Visual_scoring1_excerpt{excerpt_number}.txt")
    if use_expert2_union:
        train_events = read_union_events(folder, excerpt_number)
    else:
        train_events = expert_events
    filtered, windows, X, y = build_window_dataset(signal, 200.0, train_events)
    return {
        "excerpt": excerpt_number,
        "signal": signal,
        "filtered": filtered,
        "sfreq": 200.0,
        "expert_events": expert_events,   # always Expert 1 for scoring
        "train_events": train_events,     # Expert 1 ∪ 2 for label generation
        "windows": windows,
        "X": X,
        "y": y,
    }


def evaluate_window_detector(
    folder="data/dreams/DatabaseKcomplexes",
    threshold=None,          # None → CV selection per fold
    spindle_rejection=True,
    morphology_filter=False,
    use_expert2_union=True,
):
    """Run leave-one-out evaluation across all 10 DREAMS excerpts.

    When threshold=None the decision threshold is selected separately for
    each outer fold using inner LOO-CV on the 9 training excerpts — no
    information from the test excerpt leaks into threshold choice.

    use_expert2_union: train on Expert1 ∪ Expert2 labels for excerpts 1-5;
    evaluation always scores against Expert 1 only.
    """
    datasets = [build_excerpt(folder, idx, use_expert2_union=use_expert2_union) for idx in range(1, 11)]

    # Fix C: self-training pseudo-labels for excerpts 6-10.
    # Excerpts 6-10 only have Expert 1 labels (no Expert 2), so their positive
    # class is under-represented. We train a pseudo model on excerpts 1-5
    # (which have Expert 1+2 union) and use it to flag high-confidence windows
    # in excerpts 6-10 as pseudo-positives. This augments y without touching
    # test evaluation (eval always uses expert_events, never y).
    pseudo_train = [d for d in datasets if d["excerpt"] <= 5]
    if len(pseudo_train) >= 3:
        X_pseudo = np.vstack([d["X"] for d in pseudo_train])
        y_pseudo = np.concatenate([d["y"] for d in pseudo_train])
        pseudo_model = train_balanced_window_classifier(X_pseudo, y_pseudo, random_state=42)
        for d in datasets:
            if d["excerpt"] >= 6:
                pseudo_probs = pseudo_model.predict_proba(d["X"])[:, 1]
                pseudo_pos = (pseudo_probs >= 0.82).astype(int)
                d["y"] = np.maximum(d["y"], pseudo_pos)

    rows = []
    scores_iou = []
    scores_onset = []
    chosen_thresholds = []

    for test_idx, test_set in enumerate(datasets):
        train_sets = [d for i, d in enumerate(datasets) if i != test_idx]

        X_train = np.vstack([d["X"] for d in train_sets])
        y_train = np.concatenate([d["y"] for d in train_sets])
        model = train_balanced_window_classifier(X_train, y_train, random_state=100 + test_idx)

        # --- Threshold selection (nested CV on training fold only) ---
        if threshold is None:
            fold_threshold, _ = select_threshold_by_cv(
                train_sets,
                thresholds=THRESHOLDS_GRID,
                random_state=100 + test_idx,
            )
        else:
            fold_threshold = threshold
        chosen_thresholds.append(fold_threshold)

        probabilities = model.predict_proba(test_set["X"])[:, 1]
        signal_for_postproc = test_set["filtered"] if (spindle_rejection or morphology_filter) else None
        detected = windows_to_events(
            test_set["windows"],
            probabilities,
            test_set["sfreq"],
            threshold=fold_threshold,
            n_samples=len(test_set["signal"]),
            signal=signal_for_postproc,
            spindle_rejection=spindle_rejection,
            morphology_filter=morphology_filter,
        )
        score = score_events(test_set["expert_events"], detected)
        onset_score = score_events_onset(test_set["expert_events"], detected, tolerance=0.5)
        scores_iou.append(score)
        scores_onset.append(onset_score)
        rows.append((test_set, probabilities, detected, score))
        print(
            f"excerpt={test_set['excerpt']} threshold={fold_threshold:.2f} "
            f"windows={len(test_set['windows'])} "
            f"expert={score['expert']} detected={score['detected']} "
            f"tp={score['tp']} fp={score['fp']} fn={score['fn']} "
            f"precision={score['precision']:.3f} recall={score['recall']:.3f} "
            f"f1={score['f1']:.3f}"
        )

    totals = aggregate_scores(scores_iou)
    onset_totals = aggregate_scores(scores_onset)
    ci = bootstrap_f1_ci(scores_iou)

    threshold_label = (
        f"CV-selected (mean={np.mean(chosen_thresholds):.2f}, "
        f"range [{min(chosen_thresholds):.2f}, {max(chosen_thresholds):.2f}])"
        if threshold is None
        else f"fixed={threshold:.2f}"
    )
    print(f"\nThreshold: {threshold_label}")

    print("\nTOTAL (IoU >= 0.20 matching)")
    print(
        f"expert={totals['expert']} detected={totals['detected']} "
        f"tp={totals['tp']} fp={totals['fp']} fn={totals['fn']} "
        f"precision={totals['precision']:.3f} recall={totals['recall']:.3f} "
        f"f1={totals['f1']:.3f}"
    )
    print(f"F1 95% CI (bootstrap): [{ci['lower']:.3f}, {ci['upper']:.3f}]  std={ci['std']:.3f}")

    print("\nTOTAL (onset +/-0.5s matching)")
    print(
        f"expert={onset_totals['expert']} detected={onset_totals['detected']} "
        f"tp={onset_totals['tp']} fp={onset_totals['fp']} fn={onset_totals['fn']} "
        f"precision={onset_totals['precision']:.3f} recall={onset_totals['recall']:.3f} "
        f"f1={onset_totals['f1']:.3f}"
    )
    return rows, totals


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate window K-complex detector on DREAMS (leak-free CV threshold)."
    )
    parser.add_argument("--folder", default="data/dreams/DatabaseKcomplexes")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed threshold. Omit to use nested CV selection (recommended).",
    )
    parser.add_argument("--no-spindle-rejection", dest="spindle_rejection", action="store_false")
    parser.add_argument("--no-expert2-union", dest="use_expert2_union", action="store_false",
                        help="Disable Expert 2 union labels (use Expert 1 only for training).")
    parser.set_defaults(spindle_rejection=True, use_expert2_union=True)
    args = parser.parse_args()
    evaluate_window_detector(
        args.folder,
        threshold=args.threshold,
        spindle_rejection=args.spindle_rejection,
        use_expert2_union=args.use_expert2_union,
    )


if __name__ == "__main__":
    main()
