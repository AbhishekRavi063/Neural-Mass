from benchmarks.dreams_window_detector import evaluate_window_detector


MIN_DREAMS_F1 = 0.62


def main():
    _, totals = evaluate_window_detector(
        "data/dreams/DatabaseKcomplexes",
        threshold=0.50,
    )
    print("\nBenchmark gate")
    print(f"- required F1: {MIN_DREAMS_F1:.3f}")
    print(f"- observed F1: {totals['f1']:.3f}")
    if totals["f1"] < MIN_DREAMS_F1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
