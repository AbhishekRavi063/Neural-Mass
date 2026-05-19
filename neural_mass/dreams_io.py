"""Readers for the DREAMS K-complex database text files.

Used by benchmarks and tests. Kept minimal — no plotting, no detection logic.
"""
from pathlib import Path

import numpy as np


def read_scoring_file(path) -> list[dict]:
    """Parse a DREAMS Visual_scoring*.txt annotation file.

    Each non-header line contains: onset_seconds  duration_seconds

    Returns
    -------
    list of dicts with keys: onset, end, duration
    """
    events = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        onset = float(parts[0])
        duration = float(parts[1])
        events.append({"onset": onset, "end": onset + duration, "duration": duration})
    return events


def read_signal_txt(path) -> np.ndarray:
    """Parse a DREAMS excerpt*.txt signal file (one sample per line).

    Returns
    -------
    ndarray, shape (n_samples,)
    """
    values = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        values.append(float(line))
    return np.asarray(values)
