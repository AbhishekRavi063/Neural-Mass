"""Sklearn-style sleep event detectors."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KComplexDetector:
    """K-complex detector with sklearn-style fit/predict API.

    Trains a balanced random forest on candidate windows extracted from
    labeled EEG excerpts. Optionally applies spindle rejection post-processing
    to reduce false positives.

    Parameters
    ----------
    threshold : float
        Classification probability threshold (default: 0.50).
    sfreq : float
        EEG sampling frequency in Hz (default: 200.0).
    spindle_rejection : bool
        Reject windows dominated by sigma-band (spindle) energy (default: True).

    Examples
    --------
    >>> detector = KComplexDetector(threshold=0.50)
    >>> detector.fit(train_signals, train_expert_events)
    >>> events = detector.predict(test_signal)
    >>> scores = detector.score(test_signal, expert_events)
    >>> print(scores["f1"])
    """

    def __init__(
        self,
        threshold: float = 0.70,
        sfreq: float = 200.0,
        spindle_rejection: bool = True,
    ):
        self.threshold = threshold
        self.sfreq = sfreq
        self.spindle_rejection = spindle_rejection
        self._model = None

    def fit(
        self,
        signals: list[NDArray],
        expert_events_list: list[list[dict]],
    ) -> "KComplexDetector":
        """Train detector on labeled EEG excerpts.

        Parameters
        ----------
        signals : list of arrays, each shape (n_samples,)
        expert_events_list : list of lists of dicts (onset, end keys)

        Returns
        -------
        self
        """
        from kcomplex_detector.kcomplex_window_detector import (
            build_window_dataset,
            train_balanced_window_classifier,
        )

        all_X, all_y = [], []
        for signal, expert_events in zip(signals, expert_events_list):
            _, _, X, y = build_window_dataset(signal, self.sfreq, expert_events)
            all_X.append(X)
            all_y.append(y)
        self._model = train_balanced_window_classifier(
            np.vstack(all_X), np.concatenate(all_y)
        )
        return self

    def predict(self, signal: NDArray) -> list[dict]:
        """Detect K-complex events in a signal.

        Returns
        -------
        list of dicts with keys: onset, end, duration
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        from kcomplex_detector.kcomplex_window_detector import (
            build_window_dataset,
            windows_to_events,
        )

        signal = np.asarray(signal, dtype=float)
        # build_window_dataset internally bandpass-filters; reuse its filtered output
        # instead of filtering a second time.
        filtered, windows, X, _ = build_window_dataset(signal, self.sfreq, [])
        probabilities = self._model.predict_proba(X)[:, 1]
        return windows_to_events(
            windows,
            probabilities,
            self.sfreq,
            threshold=self.threshold,
            n_samples=len(signal),
            signal=filtered if self.spindle_rejection else None,
            spindle_rejection=self.spindle_rejection,
        )

    def score(self, signal: NDArray, expert_events: list[dict]) -> dict:
        """Predict and score against expert events using IoU matching.

        Returns
        -------
        dict with keys: expert, detected, tp, fp, fn, precision, recall, f1
        """
        from kcomplex_detector.utils.event_scoring import score_events

        return score_events(expert_events, self.predict(signal))

    def score_onset(
        self,
        signal: NDArray,
        expert_events: list[dict],
        tolerance: float = 0.5,
    ) -> dict:
        """Score using onset-proximity matching (|onset_a - onset_b| <= tolerance)."""
        from kcomplex_detector.utils.event_scoring import score_events_onset

        return score_events_onset(expert_events, self.predict(signal), tolerance=tolerance)


class SpindleDetector:
    """Threshold-based sleep spindle detector.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz (default: 200.0).
    threshold_std : float
        RMS threshold in standard deviations above the median (default: 1.5).
    min_duration : float
        Minimum spindle duration in seconds (default: 0.5).

    Examples
    --------
    >>> detector = SpindleDetector(sfreq=200)
    >>> mask = detector.predict(eeg_signal)
    """

    def __init__(
        self,
        sfreq: float = 200.0,
        threshold_std: float = 1.5,
        min_duration: float = 0.5,
    ):
        self.sfreq = sfreq
        self.threshold_std = threshold_std
        self.min_duration = min_duration

    def predict(self, signal: NDArray) -> NDArray:
        """Return boolean mask of detected spindle regions."""
        from kcomplex_detector.event_detection import spindle_detection

        return spindle_detection(
            np.asarray(signal, dtype=float),
            sampling_frequency=int(self.sfreq),
            threshold_std=self.threshold_std,
            min_duration=self.min_duration,
        )

    def predict_events(self, signal: NDArray) -> list[dict]:
        """Return detected spindle events as a list of dicts (onset, end, duration).

        Provides the same interface as KComplexDetector.predict() for uniform
        downstream processing.
        """
        from kcomplex_detector.event_detection import mask_segments

        mask = self.predict(signal)
        events = []
        for start, end in mask_segments(mask):
            onset = start / self.sfreq
            end_time = (end + 1) / self.sfreq
            events.append({
                "onset": onset,
                "end": end_time,
                "duration": end_time - onset,
            })
        return events
