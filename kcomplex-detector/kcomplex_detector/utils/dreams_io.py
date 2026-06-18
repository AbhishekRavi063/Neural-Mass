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
        try:
            onset = float(parts[0])
            duration = float(parts[1])
        except ValueError:
            continue
        if duration <= 0:
            continue
        events.append({"onset": onset, "end": onset + duration, "duration": duration})
    return events


def read_union_events(folder, excerpt_number: int) -> list[dict]:
    """Return Expert 1 ∪ Expert 2 events for excerpts 1-5; Expert 1 only otherwise.

    Union is formed by merging non-overlapping Expert 2 events into Expert 1's list.
    Used for training labels only — evaluation always scores against Expert 1 alone.
    """
    folder = Path(folder)
    e1 = read_scoring_file(folder / f"Visual_scoring1_excerpt{excerpt_number}.txt")
    e2_path = folder / f"Visual_scoring2_excerpt{excerpt_number}.txt"
    if not e2_path.exists():
        return e1
    e2 = read_scoring_file(e2_path)
    merged = list(e1)
    for ev in e2:
        # Only add Expert 2 event if it doesn't overlap any Expert 1 event
        overlap = any(
            min(ev["end"], ref["end"]) - max(ev["onset"], ref["onset"]) > 0
            for ref in e1
        )
        if not overlap:
            merged.append(ev)
    merged.sort(key=lambda e: e["onset"])
    return merged


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
        try:
            values.append(float(line))
        except ValueError:
            continue
    return np.asarray(values)
