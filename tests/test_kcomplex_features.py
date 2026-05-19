import numpy as np

from neural_mass.kcomplex_features import (
    extract_kcomplex_features,
    generate_kcomplex_candidates,
    generate_multitaper_kcomplex_candidates,
    label_candidates,
    teager_energy,
)


def test_teager_energy_shape():
    x = np.array([0.0, 1.0, 0.0, -1.0, 0.0])

    energy = teager_energy(x)

    assert energy.shape == (3,)
    assert np.isfinite(energy).all()


def test_extract_features_returns_finite_vector():
    sfreq = 100
    signal = np.zeros(500)
    signal[200:260] = -50
    event = {"start": 190, "end": 280, "onset": 1.9, "end_time": 2.8, "duration": 0.9}

    features = extract_kcomplex_features(signal, event, sfreq)

    assert features.ndim == 1
    assert len(features) >= 20
    assert np.isfinite(features).all()


def test_label_candidates_matches_expert_by_iou():
    candidates = [
        {"onset": 0.0, "end_time": 1.0},
        {"onset": 5.0, "end_time": 6.0},
    ]
    expert = [{"onset": 0.2, "end": 1.1}]

    labels = label_candidates(candidates, expert, iou_threshold=0.2)

    assert labels.tolist() == [1, 0]


def test_generate_candidates_returns_events_for_large_wave():
    sfreq = 100
    signal = np.zeros(1000)
    signal[400:460] = -80

    candidates = generate_kcomplex_candidates(signal, sfreq)

    assert len(candidates) >= 1


def test_multitaper_candidates_returns_events_for_large_slow_wave():
    sfreq = 100
    signal = np.zeros(1000)
    t = np.arange(120) / sfreq
    signal[400:520] = -50 * np.exp(-((t - 0.35) ** 2) / 0.02) + 30 * np.exp(-((t - 0.8) ** 2) / 0.03)

    candidates = generate_multitaper_kcomplex_candidates(signal, sfreq, score_threshold=0.8)

    assert len(candidates) >= 1
