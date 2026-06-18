import numpy as np


def event_iou(a, b):
    overlap = max(0.0, min(a["end"], b["end"]) - max(a["onset"], b["onset"]))
    union = max(a["end"], b["end"]) - min(a["onset"], b["onset"])
    return overlap / union if union > 0 else 0.0


def score_events(expert_events, detected_events, iou_threshold=0.2):
    matched_detected = set()
    true_positive = 0
    overlaps = []
    onset_errors = []
    duration_errors = []

    for expert in expert_events:
        best_idx = None
        best_iou = 0.0
        for idx, detected in enumerate(detected_events):
            if idx in matched_detected:
                continue
            iou = event_iou(expert, detected)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            true_positive += 1
            matched_detected.add(best_idx)
            overlaps.append(best_iou)
            matched = detected_events[best_idx]
            onset_errors.append(abs(expert["onset"] - matched["onset"]))
            duration_errors.append(abs((expert["end"] - expert["onset"]) - (matched["end"] - matched["onset"])))

    false_positive = len(detected_events) - len(matched_detected)
    false_negative = len(expert_events) - true_positive
    precision = true_positive / len(detected_events) if detected_events else 0.0
    recall = true_positive / len(expert_events) if expert_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "expert": len(expert_events),
        "detected": len(detected_events),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_onset_error": float(np.mean(onset_errors)) if onset_errors else 0.0,
        "mean_duration_error": float(np.mean(duration_errors)) if duration_errors else 0.0,
    }


def score_events_onset(expert_events, detected_events, tolerance=0.5):
    """Score events using onset-proximity matching.

    A detection matches an expert event if |onset_detected - onset_expert| <= tolerance.
    This is less sensitive to duration disagreements than IoU and is the standard
    criterion in several published K-complex papers.
    """
    matched_detected = set()
    true_positive = 0
    onset_errors = []

    for expert in expert_events:
        best_idx = None
        best_distance = float("inf")
        for idx, detected in enumerate(detected_events):
            if idx in matched_detected:
                continue
            distance = abs(expert["onset"] - detected["onset"])
            if distance < best_distance:
                best_distance = distance
                best_idx = idx
        if best_idx is not None and best_distance <= tolerance:
            true_positive += 1
            matched_detected.add(best_idx)
            onset_errors.append(best_distance)

    false_positive = len(detected_events) - len(matched_detected)
    false_negative = len(expert_events) - true_positive
    precision = true_positive / len(detected_events) if detected_events else 0.0
    recall = true_positive / len(expert_events) if expert_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "expert": len(expert_events),
        "detected": len(detected_events),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_onset_error": float(np.mean(onset_errors)) if onset_errors else 0.0,
    }


def bootstrap_f1_ci(per_excerpt_scores, n_bootstrap=1000, seed=0):
    """Bootstrap 95% CI for aggregate F1 by resampling excerpts with replacement.

    Parameters
    ----------
    per_excerpt_scores : list of score dicts (each with tp, fp, fn keys)
    n_bootstrap : int
    seed : int

    Returns
    -------
    dict with mean, lower (2.5%), upper (97.5%), std
    """
    rng = np.random.default_rng(seed)
    n = len(per_excerpt_scores)
    f1_samples = []
    for _ in range(n_bootstrap):
        sampled = [per_excerpt_scores[i] for i in rng.integers(0, n, size=n)]
        total = aggregate_scores(sampled)
        f1_samples.append(total["f1"])
    f1_arr = np.array(f1_samples)
    return {
        "mean": float(np.mean(f1_arr)),
        "lower": float(np.percentile(f1_arr, 2.5)),
        "upper": float(np.percentile(f1_arr, 97.5)),
        "std": float(np.std(f1_arr)),
    }


def aggregate_scores(scores):
    expert = sum(score["expert"] for score in scores)
    detected = sum(score["detected"] for score in scores)
    tp = sum(score["tp"] for score in scores)
    fp = sum(score["fp"] for score in scores)
    fn = sum(score["fn"] for score in scores)
    precision = tp / detected if detected else 0.0
    recall = tp / expert if expert else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expert": expert,
        "detected": detected,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(np.mean([score.get("mean_iou", 0.0) for score in scores if score.get("tp", 0) > 0])) if any(score.get("tp", 0) > 0 for score in scores) else 0.0,
        "mean_onset_error": float(np.mean([score.get("mean_onset_error", 0.0) for score in scores if score.get("tp", 0) > 0])) if any(score.get("tp", 0) > 0 for score in scores) else 0.0,
        "mean_duration_error": float(np.mean([score.get("mean_duration_error", 0.0) for score in scores if score.get("tp", 0) > 0])) if any(score.get("tp", 0) > 0 for score in scores) else 0.0,
    }
