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

from neural_mass._detection import KComplexDetector, SpindleDetector
from neural_mass._models import JansenRitModel, ThalamocorticalSleepModel

__version__ = "0.1.0"

__all__ = [
    "JansenRitModel",
    "ThalamocorticalSleepModel",
    "KComplexDetector",
    "SpindleDetector",
    "__version__",
]
