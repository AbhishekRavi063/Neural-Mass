"""Neural mass modeling toolkit."""

from src.event_detection import K_complex_detection, spindle_detection
from src.graph import ComputationalGraph, Connection, Population
from src.inference import find_best_parameters, run_simulation
from src.metrics import (
    calculate_correlation,
    calculate_rhythmicity,
    calculate_rmse,
    calculate_snr,
    calculate_spectral_snr,
    get_performance_report,
)
from src.kcomplex_window_detector import (
    build_window_dataset,
    train_balanced_window_classifier,
    windows_to_events,
)
from src.thalamocortical_model import (
    ThalamocorticalModel,
    ThalamocorticalParameters,
    simulate_thalamocortical_sleep,
)
from src.thalamocortical_fitting import (
    build_condition_summary,
    extract_window_features,
    fit_thalamocortical_features,
    fit_thalamocortical_waveform,
)

__all__ = [
    "ComputationalGraph",
    "Connection",
    "Population",
    "K_complex_detection",
    "spindle_detection",
    "find_best_parameters",
    "run_simulation",
    "calculate_correlation",
    "calculate_rhythmicity",
    "calculate_rmse",
    "calculate_snr",
    "calculate_spectral_snr",
    "get_performance_report",
    "ThalamocorticalModel",
    "ThalamocorticalParameters",
    "simulate_thalamocortical_sleep",
    "build_condition_summary",
    "extract_window_features",
    "fit_thalamocortical_features",
    "fit_thalamocortical_waveform",
    "build_window_dataset",
    "train_balanced_window_classifier",
    "windows_to_events",
]
