from dreams_kcomplex_classifier import evaluate_classifier


MIN_DREAMS_F1 = 0.58


def main():
    _, totals = evaluate_classifier(
        "data/dreams/DatabaseKcomplexes",
        threshold=0.75,
        model_name="logistic",
        candidate_method="hybrid",
    )
    print("\nBenchmark gate")
    print(f"- required F1: {MIN_DREAMS_F1:.3f}")
    print(f"- observed F1: {totals['f1']:.3f}")
    if totals["f1"] < MIN_DREAMS_F1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
