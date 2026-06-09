"""Neural mass model library for sleep EEG analysis.

Quick start
-----------
Simulate NREM-like EEG::

    from neural_mass import ThalamocorticalSleepModel
    model = ThalamocorticalSleepModel(neuromodulator_level=1.0, seed=7)
    signals = model.simulate(seconds=30.0)

Detect K-complexes::

    from neural_mass import KComplexDetector
    detector = KComplexDetector()
    detector.fit(train_signals, train_expert_events)
    events = detector.predict(test_signal)
    print(detector.score(test_signal, expert_events)["f1"])

Fit Jansen-Rit parameters::

    from neural_mass import JansenRitModel
    model = JansenRitModel()
    model.fit(target_eeg, n_trials=100)
    print(model.A_, model.B_)
"""

from neural_mass.detection._detection import KComplexDetector, SpindleDetector
from neural_mass.models._models import JansenRitModel, ThalamocorticalSleepModel, JensenRitModel
from neural_mass.inference.thalamocortical_fitting import (
    extract_profile_features,
    fit_thalamocortical_multi_objective,
    fit_schizophrenia,
    fit_depression,
)
from neural_mass.models.thalamocortical_model import build_neuromodulator_schedule, simulate_thalamocortical_sleep
from neural_mass.utils.preprocessing import preprocess_eeg, adaptive_scale, clinical_artifact_filter
from neural_mass.models.spatiotemporal_model import SpatiotemporalThalamocorticalModel
from neural_mass.detection.clinical import compute_so_pac, estimate_thalamic_gating

__version__ = "0.1.0"

__all__ = [
    "JansenRitModel",
    "JensenRitModel",
    "ThalamocorticalSleepModel",
    "SpatiotemporalThalamocorticalModel",
    "KComplexDetector",
    "SpindleDetector",
    "extract_profile_features",
    "fit_thalamocortical_multi_objective",
    "fit_schizophrenia",
    "fit_depression",
    "build_neuromodulator_schedule",
    "simulate_thalamocortical_sleep",
    "preprocess_eeg",
    "adaptive_scale",
    "clinical_artifact_filter",
    "compute_so_pac",
    "estimate_thalamic_gating",
    "__version__",
]
