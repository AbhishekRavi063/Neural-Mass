"""Sklearn-style wrappers for neural mass simulation and fitting."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class JansenRitModel:
    """Jansen-Rit cortical neural mass model.

    Simulates EEG-like population activity and fits synaptic gain parameters
    to a target signal using Optuna TPE search.

    Parameters
    ----------
    A : float
        Excitatory synaptic gain (default: 3.25).
    B : float
        Inhibitory synaptic gain (default: 22.0).
    dt : float
        Integration time step in seconds (default: 0.001).
    seed : int or None
        Random seed for reproducibility.

    Examples
    --------
    >>> model = JansenRitModel()
    >>> eeg = model.simulate(seconds=10.0)
    >>> model.fit(target_signal, n_trials=100)
    >>> print(model.A_, model.B_)
    """

    def __init__(
        self,
        A: float = 3.25,
        B: float = 22.0,
        dt: float = 0.001,
        seed: int | None = None,
    ):
        self.A = A
        self.B = B
        self.dt = dt
        self.seed = seed
        self.A_: float | None = None
        self.B_: float | None = None
        self.fit_error_: float | None = None

    def simulate(self, seconds: float = 30.0) -> NDArray:
        """Simulate EEG-like cortical output.

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        from neural_mass.graph import ComputationalGraph, Connection, Population

        cortex = Population(A=self.A, B=self.B)
        thalamus = Population(A=self.A, B=self.B)
        connections = [
            Connection(cortex, thalamus, weight=10.0),
            Connection(thalamus, cortex, weight=10.0),
        ]
        graph = ComputationalGraph(
            [cortex, thalamus], connections, dt=self.dt, seed=self.seed
        )
        return graph.simulate(seconds=seconds)[:, 0]

    def fit(self, signal: NDArray, n_trials: int = 50) -> "JansenRitModel":
        """Fit A and B parameters to minimize RMSE against a target signal.

        Parameters
        ----------
        signal : array-like, shape (n_samples,)
        n_trials : int

        Returns
        -------
        self
        """
        from neural_mass.inference import find_best_parameters

        params, error = find_best_parameters(
            signal, n_trials=n_trials, seed=self.seed or 42
        )
        self.A = params["A"]
        self.B = params["B"]
        self.A_ = params["A"]
        self.B_ = params["B"]
        self.fit_error_ = error
        return self

    @property
    def parameters_(self) -> dict:
        """Fitted parameters — available after fit()."""
        if self.A_ is None:
            raise RuntimeError("Call fit() first.")
        return {"A": self.A_, "B": self.B_, "rmse": self.fit_error_}


class ThalamocorticalSleepModel:
    """Compact thalamocortical sleep model.

    Simulates NREM-like EEG with cortical slow oscillations and thalamic
    spindle-band rhythms. The neuromodulator_level parameter represents
    cholinergic/noradrenergic tone and shifts the model between sleep stages.

    Parameters
    ----------
    neuromodulator_level : float
        0 = wake / REM (high ACh/NE, tonic relay mode).
        0.5 = light NREM (N1/N2, moderate spindles).
        1.0 = deep NREM / SWS (low ACh/NE, burst mode, slow waves).
    seed : int or None

    Examples
    --------
    >>> nrem = ThalamocorticalSleepModel(neuromodulator_level=1.0, seed=7)
    >>> signals = nrem.simulate(seconds=30.0)
    >>> print(signals["eeg"].shape)
    """

    def __init__(
        self,
        neuromodulator_level: float = 0.0,
        seed: int | None = None,
    ):
        self.neuromodulator_level = neuromodulator_level
        self.seed = seed
        self.parameters_: dict | None = None
        self.fit_error_: float | None = None

    def simulate(
        self, seconds: float = 30.0, sampling_frequency: int = 200
    ) -> dict[str, NDArray]:
        """Simulate NREM-like EEG signals.

        Returns
        -------
        dict with keys: eeg, cortical_pyramidal, thalamic_relay,
        thalamic_reticular, cortical_interneuron, adaptation, spindle
        """
        from neural_mass.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters

        dt = 0.001
        params = ThalamocorticalParameters(
            dt=dt,
            neuromodulator_level=self.neuromodulator_level,
        )
        model = ThalamocorticalModel(params, seed=self.seed)
        raw = model.simulate(seconds=seconds)
        stride = max(1, int(round((1 / dt) / sampling_frequency)))
        return {name: values[::stride] for name, values in raw.items()}

    def fit(
        self,
        signal: NDArray,
        sfreq: int = 200,
        n_trials: int = 60,
    ) -> "ThalamocorticalSleepModel":
        """Fit model parameters to window-level EEG features.

        Parameters
        ----------
        signal : array-like, shape (n_samples,)
        sfreq : int
        n_trials : int

        Returns
        -------
        self
        """
        from dataclasses import asdict

        from neural_mass.thalamocortical_fitting import (
            extract_window_features,
            fit_thalamocortical_features,
        )

        signal = np.asarray(signal, dtype=float)
        seconds = len(signal) / sfreq
        target_features = extract_window_features(signal, sfreq)
        params, _, error = fit_thalamocortical_features(
            target_features,
            seconds=seconds,
            sfreq=sfreq,
            n_trials=n_trials,
            seed=self.seed or 42,
        )
        self.parameters_ = asdict(params)
        self.fit_error_ = error
        return self
