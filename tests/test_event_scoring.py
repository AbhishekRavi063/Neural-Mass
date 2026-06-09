from neural_mass.utils.event_scoring import aggregate_scores, event_iou, score_events


def test_event_iou_for_partial_overlap():
    first = {"onset": 0.0, "end": 2.0}
    second = {"onset": 1.0, "end": 3.0}

    assert event_iou(first, second) == 1 / 3


def test_score_events_counts_tp_fp_fn():
    expert = [
        {"onset": 0.0, "end": 1.0},
        {"onset": 5.0, "end": 6.0},
    ]
    detected = [
        {"onset": 0.1, "end": 1.1},
        {"onset": 10.0, "end": 11.0},
    ]

    scores = score_events(expert, detected, iou_threshold=0.2)

    assert scores["tp"] == 1
    assert scores["fp"] == 1
    assert scores["fn"] == 1
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["mean_onset_error"] == 0.1
    assert round(scores["mean_duration_error"], 6) == 0.0


def test_aggregate_scores_micro_averages_counts():
    scores = [
        {"expert": 2, "detected": 2, "tp": 1, "fp": 1, "fn": 1},
        {"expert": 3, "detected": 1, "tp": 1, "fp": 0, "fn": 2},
    ]

    total = aggregate_scores(scores)

    assert total["expert"] == 5
    assert total["detected"] == 3
    assert total["tp"] == 2
    assert total["fp"] == 1
    assert total["fn"] == 3
    assert "mean_iou" in total
    assert "mean_onset_error" in total
    assert "mean_duration_error" in total
