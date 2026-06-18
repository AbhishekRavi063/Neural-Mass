"""Neural mass models for sleep EEG simulation and parameter fitting.

This is the simulator half of the project (the K-complex *detector* lives in the
sibling ``kcomplex-detector`` project). It provides the Jansen-Rit cortical
oscillator, a compact thalamocortical sleep model, a spatiotemporal variant, and
Optuna-based parameter-fitting routines.

Quick start
-----------
Simulate NREM-like EEG::

    from neural_mass import ThalamocorticalSleepModel
    model = ThalamocorticalSleepModel(neuromodulator_level=1.0, seed=7)
    signals = model.simulate(seconds=30.0)

Fit Jansen-Rit parameters to a target signal::

    from neural_mass import JansenRitModel
    model = JansenRitModel()
    model.fit(target_eeg, n_trials=100)
    print(model.A_, model.B_)
"""
from neural_mass.models._models import JansenRitModel, ThalamocorticalSleepModel, JensenRitModel
from neural_mass.inference.thalamocortical_fitting import (
    extract_profile_features,
    fit_thalamocortical_multi_objective,
    fit_schizophrenia,
    fit_depression,
)
from neural_mass.models.thalamocortical_model import build_neuromodulator_schedule, simulate_thalamocortical_sleep
from neural_mass.models.spatiotemporal_model import SpatiotemporalThalamocorticalModel
from neural_mass.utils.preprocessing import preprocess_eeg, adaptive_scale, clinical_artifact_filter

__version__ = "0.1.0"

__all__ = [
    "JansenRitModel",
    "JensenRitModel",
    "ThalamocorticalSleepModel",
    "SpatiotemporalThalamocorticalModel",
    "extract_profile_features",
    "fit_thalamocortical_multi_objective",
    "fit_schizophrenia",
    "fit_depression",
    "build_neuromodulator_schedule",
    "simulate_thalamocortical_sleep",
    "preprocess_eeg",
    "adaptive_scale",
    "clinical_artifact_filter",
    "__version__",
]
