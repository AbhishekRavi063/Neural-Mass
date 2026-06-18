import numpy as np

from kcomplex_detector.kcomplex_window_detector import (
    _high_freq_artifact,
    build_window_dataset,
    extract_window_features,
    label_windows,
    select_threshold_by_cv,
    sliding_windows,
    slow_wave_candidate_windows,
    windows_to_events,
)


def test_sliding_windows_uses_overlap():
    signal = np.zeros(200)

    windows = sliding_windows(signal, sfreq=100, window_seconds=0.5, overlap_seconds=0.4)

    assert windows[0] == (0, 50)
    assert windows[1] == (10, 60)
    assert len(windows) > 1


def test_extract_window_features_are_finite():
    sfreq = 100
    t = np.arange(100) / sfreq
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.2 * np.sin(2 * np.pi * 13.0 * t)

    features = extract_window_features(signal, 0, 50, sfreq)

    assert features.ndim == 1
    assert len(features) >= 20
    assert np.isfinite(features).all()


def test_label_windows_marks_overlapping_expert():
    windows = [(0, 50), (100, 150)]
    experts = [{"onset": 0.1, "end": 0.6}]

    labels = label_windows(windows, experts, sfreq=100, iou_threshold=0.1)

    assert labels.tolist() == [1, 0]


def test_windows_to_events_merges_positive_windows():
    windows = [(0, 50), (10, 60), (200, 250)]
    probabilities = np.array([0.8, 0.9, 0.1])

    events = windows_to_events(windows, probabilities, sfreq=100, threshold=0.5, n_samples=300)

    assert len(events) == 1
    assert events[0]["duration"] >= 0.5


def test_build_window_dataset_shapes():
    sfreq = 100
    signal = np.zeros(300)
    signal[100:160] = -50
    experts = [{"onset": 1.0, "end": 1.6}]

    _, windows, X, y = build_window_dataset(signal, sfreq, experts)

    assert len(windows) == len(X) == len(y)
    assert X.shape[1] >= 20


def test_slow_wave_candidate_windows_finds_large_peak():
    sfreq = 100
    signal = np.zeros(500)
    signal[200:230] = -60

    windows = slow_wave_candidate_windows(signal, sfreq, max_candidates=20)

    assert len(windows) >= 1


def test_high_freq_artifact_flags_hf_dominated_segment():
    sfreq = 200
    t = np.arange(200) / sfreq
    clean = np.sin(2 * np.pi * 1.0 * t)
    hf_burst = np.sin(2 * np.pi * 45.0 * t) * 10

    assert not _high_freq_artifact(clean, 0, 200, sfreq)
    assert _high_freq_artifact(hf_burst, 0, 200, sfreq)


def test_select_threshold_by_cv_returns_valid_threshold():
    rng = np.random.default_rng(0)
    sfreq = 100
    n_samples = 600
    datasets = []
    for i in range(4):
        signal = rng.normal(0, 0.1, n_samples)
        signal[200:260] = -15  # K-complex candidate
        filtered = signal.copy()
        # Two windows: one positive, one negative
        windows = [(190, 260), (350, 420)]
        X = np.array([extract_window_features(filtered, s, e, sfreq) for s, e in windows])
        y = np.array([1, 0])
        datasets.append({
            "X": X, "y": y, "windows": windows, "filtered": filtered,
            "sfreq": sfreq, "signal": signal,
            "expert_events": [{"onset": 2.0, "end": 2.6}],
        })

    best_t, mean_f1s = select_threshold_by_cv(datasets, thresholds=[0.3, 0.5, 0.7])

    assert best_t in [0.3, 0.5, 0.7]
    assert all(0.0 <= v <= 1.0 for v in mean_f1s.values())
