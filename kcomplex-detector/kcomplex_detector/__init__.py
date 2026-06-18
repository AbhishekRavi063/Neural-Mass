"""K-complex detection from sleep EEG.

A machine-learning pipeline that detects K-complexes in single-channel sleep
EEG: candidate-window generation -> 48-feature extraction -> gradient-boosted
classifier -> rule-based gates -> merge/duration filtering -> spindle rejection.

Quick start
-----------
    from kcomplex_detector import KComplexDetector
    detector = KComplexDetector()
    detector.fit(train_signals, train_expert_events)
    events = detector.predict(test_signal)
    print(detector.score(test_signal, expert_events)["f1"])
"""
from kcomplex_detector.detector import KComplexDetector, SpindleDetector
from kcomplex_detector.clinical import compute_so_pac, estimate_thalamic_gating
from kcomplex_detector.utils.preprocessing import (
    preprocess_eeg,
    adaptive_scale,
    clinical_artifact_filter,
)

__version__ = "0.1.0"

__all__ = [
    "KComplexDetector",
    "SpindleDetector",
    "compute_so_pac",
    "estimate_thalamic_gating",
    "preprocess_eeg",
    "adaptive_scale",
    "clinical_artifact_filter",
    "__version__",
]
