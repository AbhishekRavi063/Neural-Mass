"""Neural mass modeling toolkit."""

from src.event_detection import K_complex_detection, spindle_detection, splindle_detection
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

__all__ = [
    "ComputationalGraph",
    "Connection",
    "Population",
    "K_complex_detection",
    "spindle_detection",
    "splindle_detection",
    "find_best_parameters",
    "run_simulation",
    "calculate_correlation",
    "calculate_rhythmicity",
    "calculate_rmse",
    "calculate_snr",
    "calculate_spectral_snr",
    "get_performance_report",
]
