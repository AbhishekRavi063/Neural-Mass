import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dreams_kcomplex_validation import read_scoring_file, read_signal_txt
from src.event_scoring import aggregate_scores, score_events
from src.kcomplex_features import (
    extract_kcomplex_features,
    generate_kcomplex_candidates,
    label_candidates,
)


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1 / (1 + np.exp(-x))


def standardize_train(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


def train_logistic_regression(X, y, epochs=2500, lr=0.04, l2=0.02):
    Xs, mean, std = standardize_train(X)
    Xb = np.c_[np.ones(len(Xs)), Xs]
    weights = np.zeros(Xb.shape[1])

    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    class_weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))

    for _ in range(epochs):
        probs = sigmoid(Xb @ weights)
        error = (probs - y) * class_weights
        gradient = (Xb.T @ error) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= lr * gradient

    return {"weights": weights, "mean": mean, "std": std}


def predict_proba(model, X):
    Xs = standardize_apply(X, model["mean"], model["std"])
    Xb = np.c_[np.ones(len(Xs)), Xs]
    return sigmoid(Xb @ model["weights"])


def build_sklearn_model(model_name, y_train):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        positives = max(1, int(np.sum(y_train == 1)))
        negatives = max(1, int(np.sum(y_train == 0)))
        positive_weight = negatives / positives
        return GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            random_state=42,
        ), positive_weight
    if model_name == "logistic":
        return None
    raise ValueError(f"Unknown model: {model_name}")


def train_model(X_train, y_train, model_name):
    if model_name == "logistic":
        return train_logistic_regression(X_train, y_train), model_name

    if model_name == "gradient_boosting":
        classifier, positive_weight = build_sklearn_model(model_name, y_train)
        sample_weight = np.where(y_train == 1, positive_weight, 1.0)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), classifier)
        model.fit(X_train, y_train, gradientboostingclassifier__sample_weight=sample_weight)
        return model, model_name

    classifier = build_sklearn_model(model_name, y_train)
    model = make_pipeline(SimpleImputer(strategy="median"), classifier)
    model.fit(X_train, y_train)
    return model, model_name


def model_predict_proba(model, X, model_name):
    if model_name == "logistic":
        return predict_proba(model, X)
    return model.predict_proba(X)[:, 1]


def load_dreams_excerpt(folder, excerpt_number):
    folder = Path(folder)
    signal = read_signal_txt(folder / f"excerpt{excerpt_number}.txt")
    expert_events = read_scoring_file(folder / f"Visual_scoring1_excerpt{excerpt_number}.txt")
    return signal, 200.0, expert_events


def build_excerpt_dataset(folder, excerpt_number, candidate_method="hybrid"):
    signal, sfreq, expert_events = load_dreams_excerpt(folder, excerpt_number)
    candidates = generate_kcomplex_candidates(signal, sfreq, method=candidate_method)
    X = np.array([extract_kcomplex_features(signal, event, sfreq) for event in candidates])
    y = label_candidates(candidates, expert_events)
    return {
        "excerpt": excerpt_number,
        "signal": signal,
        "sfreq": sfreq,
        "expert_events": expert_events,
        "candidates": candidates,
        "X": X,
        "y": y,
        "candidate_method": candidate_method,
    }


def candidate_to_event(candidate):
    return {
        "onset": candidate["onset"],
        "end": candidate["end_time"],
        "duration": candidate["duration"],
    }


def threshold_events(candidates, probabilities, threshold):
    return [
        candidate_to_event(candidate)
        for candidate, probability in zip(candidates, probabilities)
        if probability >= threshold
    ]


def tune_threshold(datasets, thresholds):
    y_true = np.concatenate([dataset["y"] for dataset in datasets])
    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in thresholds:
        tp = fp = fn = 0
        offset = 0
        for dataset in datasets:
            n = len(dataset["y"])
            probs = y_true[offset:offset + n]
            offset += n
            predictions = probs >= threshold
            tp += int(np.sum((predictions == 1) & (dataset["y"] == 1)))
            fp += int(np.sum((predictions == 1) & (dataset["y"] == 0)))
            fn += int(np.sum((predictions == 0) & (dataset["y"] == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
    return best


def evaluate_classifier(folder, threshold=0.5, model_name="random_forest", candidate_method="hybrid"):
    datasets = [build_excerpt_dataset(folder, i, candidate_method=candidate_method) for i in range(1, 11)]
    rows = []
    all_scores = []

    for test_idx in range(10):
        train_sets = [dataset for idx, dataset in enumerate(datasets) if idx != test_idx]
        test_set = datasets[test_idx]
        X_train = np.vstack([dataset["X"] for dataset in train_sets])
        y_train = np.concatenate([dataset["y"] for dataset in train_sets])

        model, fitted_model_name = train_model(X_train, y_train, model_name)
        probs = model_predict_proba(model, test_set["X"], fitted_model_name)
        detected_events = threshold_events(test_set["candidates"], probs, threshold)
        scores = score_events(test_set["expert_events"], detected_events)
        all_scores.append(scores)
        rows.append((test_set, probs, detected_events, scores))

        print(
            f"model={model_name} candidates={candidate_method} excerpt={test_set['excerpt']} "
            f"candidates={len(test_set['candidates'])} "
            f"expert={scores['expert']} detected={scores['detected']} "
            f"tp={scores['tp']} fp={scores['fp']} fn={scores['fn']} "
            f"precision={scores['precision']:.3f} recall={scores['recall']:.3f} "
            f"f1={scores['f1']:.3f}"
        )

    totals = aggregate_scores(all_scores)

    print("\nTOTAL")
    print(f"expert={totals['expert']} detected={totals['detected']} tp={totals['tp']} fp={totals['fp']} fn={totals['fn']}")
    print(f"precision={totals['precision']:.3f} recall={totals['recall']:.3f} f1={totals['f1']:.3f}")
    return rows, totals


def plot_classifier_result(row, threshold, output):
    dataset, probs, detected_events, _ = row
    signal = dataset["signal"]
    sfreq = dataset["sfreq"]
    expert_events = dataset["expert_events"]
    times = np.arange(len(signal)) / sfreq

    expert_mask = np.zeros(len(signal), dtype=bool)
    detected_mask = np.zeros(len(signal), dtype=bool)
    for event in expert_events:
        expert_mask[int(event["onset"] * sfreq):int(event["end"] * sfreq)] = True
    for event in detected_events:
        detected_mask[int(event["onset"] * sfreq):int(event["end"] * sfreq)] = True

    window_start = max(0.0, expert_events[0]["onset"] - 10.0) if expert_events else 0.0
    window_end = min(times[-1], window_start + 40.0)
    window_mask = (times >= window_start) & (times <= window_end)

    plt.figure(figsize=(13, 5))
    plt.plot(times[window_mask], signal[window_mask], color="#2c3e50", linewidth=1.0, label="DREAMS EEG CZ-A1")
    y_min, y_max = np.percentile(signal[window_mask], [1, 99])
    plt.fill_between(times[window_mask], y_min, y_max, where=expert_mask[window_mask], color="#9bdbff", alpha=0.32, label="Expert")
    plt.fill_between(times[window_mask], y_min, y_max, where=detected_mask[window_mask], color="#ffb3d9", alpha=0.32, label="Classifier")
    plt.ylim(y_min, y_max)
    plt.title(f"DREAMS Classifier K-complex Validation - Excerpt {dataset['excerpt']} (threshold={threshold})")
    plt.xlabel("Time (s)")
    plt.ylabel("Potential (uV)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output)


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a simple K-complex classifier on DREAMS.")
    parser.add_argument("--folder", default="data/dreams/DatabaseKcomplexes")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--model", choices=["logistic", "random_forest", "gradient_boosting"], default="logistic")
    parser.add_argument("--candidates", choices=["rule", "multitaper", "hybrid"], default="hybrid")
    args = parser.parse_args()

    rows, totals = evaluate_classifier(
        args.folder,
        threshold=args.threshold,
        model_name=args.model,
        candidate_method=args.candidates,
    )
    best_row = max(rows, key=lambda row: row[3]["f1"])
    plot_classifier_result(best_row, args.threshold, "dreams_kcomplex_classifier.png")
    print("\nSaved dreams_kcomplex_classifier.png")


if __name__ == "__main__":
    main()
