import numpy as np


def event_iou(a, b):
    overlap = max(0.0, min(a["end"], b["end"]) - max(a["onset"], b["onset"]))
    union = max(a["end"], b["end"]) - min(a["onset"], b["onset"])
    return overlap / union if union > 0 else 0.0


def score_events(expert_events, detected_events, iou_threshold=0.2):
    matched_detected = set()
    true_positive = 0
    overlaps = []

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
    }
