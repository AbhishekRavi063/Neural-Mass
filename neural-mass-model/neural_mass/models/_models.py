"""Sklearn-style wrappers for neural mass simulation and fitting."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from neural_mass.models.thalamocortical_model import _anti_alias_and_downsample


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

        Uses fitted parameters (A_, B_) when available, otherwise constructor values.

        Returns
        -------
        ndarray, shape (n_samples,)
        """
        from neural_mass.models.graph import ComputationalGraph, Connection, Population

        A = self.A_ if self.A_ is not None else self.A
        B = self.B_ if self.B_ is not None else self.B
        cortex = Population(A=A, B=B)
        thalamus = Population(A=A, B=B)
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

        Notes
        -----
        self.A and self.B retain their original constructor values.
        Fitted values are stored in self.A_ and self.B_ (sklearn convention).
        Call simulate() after fit() to use fitted parameters automatically.
        """
        from neural_mass.inference.inference import find_best_parameters

        params, error = find_best_parameters(
            signal, n_trials=n_trials, seed=self.seed or 42
        )
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
        self.best_features_: dict | None = None
        self.fit_error_: float | None = None

    def simulate(
        self,
        seconds: float = 30.0,
        sampling_frequency: int = 200,
        neuromodulator_schedule=None,
        multi_channel: bool = False,
        pink_noise_std: float | None = None,
        micro_architecture: bool = False,
        arousals: bool = False,
    ) -> dict[str, NDArray]:
        """Simulate NREM-like EEG signals.

        Parameters
        ----------
        seconds : float
            Simulation duration.
        sampling_frequency : int
            Output sampling rate (anti-aliasing applied automatically).
        neuromodulator_schedule : array-like or None
            Per-step neuromodulator level (len = int(seconds / 0.001)).
            Enables sleep macro-architecture (stage cycling).
            Build with ``build_neuromodulator_schedule()``.
        multi_channel : bool
            If True, use a physical 3D dipole-in-sphere forward model projection to
            compute 'eeg_fz', 'eeg_cz', 'eeg_pz' (frontal, central, parietal proxies).
        pink_noise_std : float or None
            Amplitude (std) of 1/f pink noise added to the EEG output.
            Defaults to the parameters' default (0.005). Set to 0 to disable.
        micro_architecture : bool
            Enable slow parameter modulations for spindle clustering.
        arousals : bool
            Enable periodic 4-second Wake arousals with noise drive.

        Returns
        -------
        dict with keys: eeg, cortical_pyramidal, thalamic_relay,
        thalamic_reticular, cortical_interneuron, adaptation, spindle
        (plus eeg_fz/eeg_cz/eeg_pz when multi_channel=True)
        """
        from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters, _generate_pink_noise

        dt = 0.001
        params = ThalamocorticalParameters(
            dt=dt,
            neuromodulator_level=self.neuromodulator_level,
        )
        model = ThalamocorticalModel(params, seed=self.seed)
        raw = model.simulate(
            seconds=seconds,
            neuromodulator_schedule=neuromodulator_schedule,
            micro_architecture=micro_architecture,
            arousals=arousals,
        )
        out = _anti_alias_and_downsample(raw, 1 / dt, sampling_frequency)

        if multi_channel:
            # Physical Forward Model: Single-dipole-in-sphere approximation
            # Electrode coordinates on a unit sphere (R=1)
            r_fz = np.array([0.50, 0.0, 0.866])
            r_cz = np.array([0.0, 0.0, 1.0])
            r_pz = np.array([-0.50, 0.0, 0.866])

            # Cortical slow-wave radial dipole source (Frontal location)
            r_cort = np.array([0.40, 0.0, 0.70])
            d_cort = np.array([0.50, 0.0, 0.866])

            # Thalamic vertical dipole source (Deep location)
            r_thal = np.array([0.0, 0.0, 0.15])
            d_thal = np.array([0.0, 0.0, 1.0])

            # Distance vectors
            diff_fz_cort = r_fz - r_cort
            diff_cz_cort = r_cz - r_cort
            diff_pz_cort = r_pz - r_cort

            diff_fz_thal = r_fz - r_thal
            diff_cz_thal = r_cz - r_thal
            diff_pz_thal = r_pz - r_thal

            # Projections
            proj_fz_cort = np.dot(d_cort, diff_fz_cort) / (np.linalg.norm(diff_fz_cort) ** 3 + 1e-6)
            proj_cz_cort = np.dot(d_cort, diff_cz_cort) / (np.linalg.norm(diff_cz_cort) ** 3 + 1e-6)
            proj_pz_cort = np.dot(d_cort, diff_pz_cort) / (np.linalg.norm(diff_pz_cort) ** 3 + 1e-6)

            proj_fz_thal = np.dot(d_thal, diff_fz_thal) / (np.linalg.norm(diff_fz_thal) ** 3 + 1e-6)
            proj_cz_thal = np.dot(d_thal, diff_cz_thal) / (np.linalg.norm(diff_cz_thal) ** 3 + 1e-6)
            proj_pz_thal = np.dot(d_thal, diff_pz_thal) / (np.linalg.norm(diff_pz_thal) ** 3 + 1e-6)

            cortical = out["cortical_pyramidal"]
            spindle  = out["spindle"]
            relay    = out["thalamic_relay"]

            # Normalized to Cz
            p_fz_c = proj_fz_cort / (proj_cz_cort + 1e-6)
            p_cz_c = 1.0
            p_pz_c = proj_pz_cort / (proj_cz_cort + 1e-6)

            p_fz_t = proj_fz_thal / (proj_cz_thal + 1e-6)
            p_cz_t = 1.0
            p_pz_t = proj_pz_thal / (proj_cz_thal + 1e-6)

            out["eeg_fz"] = p_fz_c * cortical + params.eeg_spindle_weight * p_fz_t * spindle + params.eeg_relay_weight * p_fz_t * relay
            out["eeg_cz"] = out["eeg"]
            out["eeg_pz"] = p_pz_c * cortical + params.eeg_spindle_weight * p_pz_t * spindle + params.eeg_relay_weight * p_pz_t * relay

        # Add pink noise if specified
        if pink_noise_std is None:
            pink_noise_std = params.pink_noise_std

        if pink_noise_std > 0:
            rng = np.random.default_rng(self.seed)
            n_samples = len(out["eeg"])
            p_noise = _generate_pink_noise(n_samples, pink_noise_std, rng)
            out["eeg"] = out["eeg"] + p_noise
            if "eeg_fz" in out:
                out["eeg_fz"] = out["eeg_fz"] + p_noise
                out["eeg_cz"] = out["eeg_cz"] + p_noise
                out["eeg_pz"] = out["eeg_pz"] + p_noise

        return out

    def fit(
        self,
        signal: NDArray,
        sfreq: int = 200,
        n_trials: int = 60,
    ) -> "ThalamocorticalSleepModel":
        """Fit model parameters to target signal using multi-objective Optuna.

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

        from neural_mass.inference.thalamocortical_fitting import fit_thalamocortical_multi_objective

        signal = np.asarray(signal, dtype=float)
        params, best_features, error = fit_thalamocortical_multi_objective(
            signal,
            sfreq=sfreq,
            n_trials=n_trials,
            seed=self.seed or 42,
        )
        self.parameters_ = asdict(params)
        self.best_features_ = best_features
        self.fit_error_ = error
        return self

    @property
    def best_parameters_(self) -> dict:
        """Fitted parameters — available after fit()."""
        if self.parameters_ is None:
            raise RuntimeError("Call fit() first.")
        return self.parameters_

    @property
    def best_parameters(self) -> dict:
        """Fitted parameters — available after fit()."""
        return self.best_parameters_


# Alias used in the guide's PDF. Points to ThalamocorticalSleepModel intentionally —
# the guide repurposed this name for the full thalamocortical model.
# Do NOT alias to JansenRitModel (that is the simpler cortex-only model above).
JensenRitModel = ThalamocorticalSleepModel
