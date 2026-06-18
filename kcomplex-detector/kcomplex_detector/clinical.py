from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt, hilbert


def compute_so_pac(signal: NDArray, sfreq: float, n_bins: int = 18) -> dict:
    """Compute Phase-Amplitude Coupling (PAC) between Slow Oscillation and Spindles.

    Parameters
    ----------
    signal : array-like, shape (n_samples,)
        EEG signal.
    sfreq : float
        Sampling frequency in Hz.
    n_bins : int
        Number of phase bins (default 18, corresponding to 20-degree bins).

    Returns
    -------
    dict
        modulation_index : float
            Kullback-Leibler based Modulation Index (MI).
        preferred_phase_rad : float
            Preferred coupling phase in radians [-pi, pi].
        preferred_phase_deg : float
            Preferred coupling phase in degrees [-180, 180].
        bin_amplitudes : ndarray
            Mean spindle amplitude per phase bin.
        bin_centers : ndarray
            Phase bin centers in radians.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array.")

    # 1. Bandpass filter slow oscillation (0.5 - 2.0 Hz)
    sos_so = butter(4, [0.5, 2.0], btype="bandpass", output="sos", fs=sfreq)
    so_signal = sosfiltfilt(sos_so, signal)

    # 2. Bandpass filter spindle (11.0 - 16.0 Hz)
    sos_sp = butter(4, [11.0, 16.0], btype="bandpass", output="sos", fs=sfreq)
    sp_signal = sosfiltfilt(sos_sp, signal)

    # 3. Extract phase of Slow Oscillation using Hilbert transform
    analytic_so = hilbert(so_signal)
    so_phase = np.angle(analytic_so)  # Instantaneous phase in [-pi, pi]

    # 4. Extract amplitude of spindle activity
    analytic_sp = hilbert(sp_signal)
    sp_amplitude = np.abs(analytic_sp)

    # 5. Bin phase values from [-pi, pi] mapped to [0, 2*pi]
    so_phase_shifted = so_phase + np.pi
    bin_edges = np.linspace(0, 2.0 * np.pi, n_bins + 1)
    bin_amplitudes = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (so_phase_shifted >= bin_edges[b]) & (so_phase_shifted < bin_edges[b + 1])
        if mask.any():
            bin_amplitudes[b] = np.mean(sp_amplitude[mask])
        else:
            bin_amplitudes[b] = 0.0

    # 6. Compute Modulation Index (MI) via KL-divergence from uniform distribution
    total_amp = np.sum(bin_amplitudes)
    if total_amp > 1e-12:
        P = bin_amplitudes / total_amp
    else:
        P = np.full(n_bins, 1.0 / n_bins)

    # D_KL(P, U) = sum( P_i * log(P_i / U_i) ) where U_i = 1 / n_bins
    mi = 0.0
    for p_val in P:
        if p_val > 1e-12:
            mi += p_val * np.log(p_val)
    mi += np.log(n_bins)
    mi /= np.log(n_bins)  # normalize to [0, 1]

    # Find preferred phase bin
    preferred_bin = np.argmax(bin_amplitudes)
    preferred_phase_rad = (bin_edges[preferred_bin] + bin_edges[preferred_bin + 1]) / 2.0 - np.pi

    return {
        "modulation_index": float(mi),
        "preferred_phase_rad": float(preferred_phase_rad),
        "preferred_phase_deg": float(np.degrees(preferred_phase_rad)),
        "bin_amplitudes": bin_amplitudes,
        "bin_centers": (bin_edges[:-1] + bin_edges[1:]) / 2.0 - np.pi
    }


def estimate_thalamic_gating(simulation_outputs: dict, reticular_threshold: float = 0.65) -> float:
    """Calculate the Thalamocortical Gating Index (TGI).

    TGI is the fraction of simulation time where the reticular nucleus activity
    exceeds the threshold, shielding the cortex from sensory input.

    Parameters
    ----------
    simulation_outputs : dict
        Output dictionary from model.simulate().
    reticular_threshold : float
        Activation threshold for reticular population (default 0.65).

    Returns
    -------
    tgi : float
        Thalamocortical Gating Index in [0, 1].
    """
    if "thalamic_reticular" not in simulation_outputs:
        raise KeyError("simulation_outputs must contain 'thalamic_reticular'.")
    reticular = np.asarray(simulation_outputs["thalamic_reticular"])
    gating_mask = reticular > reticular_threshold
    return float(np.mean(gating_mask))
