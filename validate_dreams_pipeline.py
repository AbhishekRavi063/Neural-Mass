from pathlib import Path

from dreams_kcomplex_classifier import evaluate_classifier
from dreams_kcomplex_validation import aggregate_scores, evaluate_excerpt, plot_excerpt


DREAMS_FOLDER = Path("data/dreams/DatabaseKcomplexes")
MIN_CLASSIFIER_F1 = 0.58


def main():
    print("==========================================")
    print(" DREAMS K-COMPLEX VALIDATION PIPELINE")
    print("==========================================")
    print("Dataset: DREAMS K-complex database")
    print("Ground truth: Visual_scoring1 expert labels")
    print()

    print("[1/2] Rule-based detector baseline")
    all_rule_scores = []
    first_plot = None
    for excerpt_number in range(1, 11):
        signal, sfreq, expert_events, detected_events, mask, scores = evaluate_excerpt(
            DREAMS_FOLDER,
            excerpt_number,
            preset="conservative",
        )
        all_rule_scores.append(scores)
        if first_plot is None:
            first_plot = (signal, sfreq, expert_events, mask, excerpt_number)
        print(
            f"excerpt={excerpt_number} "
            f"expert={scores['expert']} detected={scores['detected']} "
            f"tp={scores['tp']} fp={scores['fp']} fn={scores['fn']} "
            f"precision={scores['precision']:.3f} recall={scores['recall']:.3f} "
            f"f1={scores['f1']:.3f} mean_iou={scores['mean_iou']:.3f}"
        )

    rule_totals = aggregate_scores(all_rule_scores)
    print("\nRule TOTAL")
    print(
        f"expert={rule_totals['expert']} detected={rule_totals['detected']} "
        f"tp={rule_totals['tp']} fp={rule_totals['fp']} fn={rule_totals['fn']}"
    )
    print(
        f"precision={rule_totals['precision']:.3f} "
        f"recall={rule_totals['recall']:.3f} f1={rule_totals['f1']:.3f}"
    )

    if first_plot is not None:
        signal, sfreq, expert_events, mask, excerpt_number = first_plot
        plot_excerpt(
            signal,
            sfreq,
            expert_events,
            mask,
            excerpt_number,
            "conservative",
            "dreams_kcomplex_validation.png",
        )
    print()

    print("[2/2] Hybrid candidate + classifier benchmark")
    _, classifier_totals = evaluate_classifier(
        DREAMS_FOLDER,
        threshold=0.75,
        model_name="logistic",
        candidate_method="hybrid",
    )

    print("\nSummary")
    print(f"- rule baseline F1: {rule_totals['f1']:.3f}")
    print(f"- classifier F1: {classifier_totals['f1']:.3f}")
    print(f"- required classifier F1: {MIN_CLASSIFIER_F1:.3f}")

    if classifier_totals["f1"] < MIN_CLASSIFIER_F1:
        raise SystemExit(1)

    print("\nPASS: DREAMS validation gate met.")


if __name__ == "__main__":
    main()
