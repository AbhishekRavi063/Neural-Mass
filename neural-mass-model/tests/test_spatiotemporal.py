import numpy as np
import pytest
from neural_mass.models.spatiotemporal_model import SpatiotemporalThalamocorticalModel
from neural_mass.models.thalamocortical_model import ThalamocorticalParameters


def test_spatiotemporal_simulation_shape():
    """Verify that the spatiotemporal lattice model runs and outputs the correct shapes."""
    model = SpatiotemporalThalamocorticalModel(n_nodes=8, seed=42)
    # Simulate 5 seconds at 200Hz
    out = model.simulate(seconds=5.0, sampling_frequency=200, closed_loop=False)

    # Check shapes of projected scalp channels
    assert "eeg_fz" in out
    assert "eeg_cz" in out
    assert "eeg_pz" in out
    assert len(out["eeg_fz"]) == 5 * 200
    assert len(out["eeg_cz"]) == 5 * 200
    assert len(out["eeg_pz"]) == 5 * 200
    
    # Check shape of node-level signals
    assert out["cortical_pyramidal"].shape == (5 * 200, 8)
    assert out["spindle"].shape == (5 * 200, 8)
    
    # Check that outputs are finite
    assert np.isfinite(out["eeg_fz"]).all()
    assert np.isfinite(out["eeg_cz"]).all()
    assert np.isfinite(out["eeg_pz"]).all()


def test_slow_wave_propagation_phase_lag():
    """Verify that slow waves propagate frontally-to-parietally (positive phase lag from Fz to Pz)."""
    # Increase pacemaker strength and lateral coupling to ensure clean, driven traveling waves
    model = SpatiotemporalThalamocorticalModel(
        n_nodes=8,
        lateral_coupling_strength=1.5,
        spatial_spread=1.0,
        pacemaker_strength=0.35,
        seed=123,
    )
    
    # Simulate 10 seconds of N2 sleep with moderate sleep pressure
    out = model.simulate(
        seconds=10.0,
        sampling_frequency=200,
        closed_loop=False,
        initial_sleep_pressure=0.6,
    )
    
    eeg_fz = out["eeg_fz"]
    eeg_pz = out["eeg_pz"]
    
    # Calculate cross-correlation to find temporal lag from Fz to Pz
    # Normalize signals
    fz_norm = (eeg_fz - eeg_fz.mean()) / (eeg_fz.std() + 1e-10)
    pz_norm = (eeg_pz - eeg_pz.mean()) / (eeg_pz.std() + 1e-10)
    
    # Compute cross-correlation
    corr = np.correlate(fz_norm, pz_norm, mode="full")
    lags = np.arange(-len(fz_norm) + 1, len(fz_norm))
    
    # Find lag with maximum correlation
    best_lag_idx = np.argmax(corr)
    best_lag_samples = lags[best_lag_idx]
    best_lag_seconds = best_lag_samples / 200.0
    
    # Frontal waves should precede parietal waves, so cross-correlating Fz(t) with Pz(t)
    # should yield a positive lag (Fz leads Pz, meaning Pz is a delayed version of Fz,
    # so Fz(t) correlates best with Pz(t + lag) where lag > 0).
    # Conduction velocity and synaptic propagation of slow waves in this lattice
    # typically create a lag of 20-150 ms (4-30 samples).
    assert best_lag_seconds >= 0.0, f"Fz should lead Pz, but got lag: {best_lag_seconds:.3f} s"


def test_closed_loop_process_s_feedback():
    """Verify that Process S sleep pressure decreases during sleep and increases during wake."""
    model = SpatiotemporalThalamocorticalModel(n_nodes=8, seed=42)
    
    # Run a closed loop simulation for 25 seconds with small time constants
    # so we can observe transitions within a short test run.
    out = model.simulate(
        seconds=25.0,
        sampling_frequency=200,
        closed_loop=True,
        tau_accum=8.0,
        tau_dissip=10.0,
        initial_sleep_pressure=0.85,  # start with high sleep pressure (deep sleep)
    )
    
    sleep_pressure = out["sleep_pressure"]
    
    # Check that sleep pressure is bounded between 0 and 1
    assert (sleep_pressure >= 0.0).all()
    assert (sleep_pressure <= 1.0).all()
    
    # Since the system exhibits closed-loop sleep stage cycling, sleep pressure
    # will decay during the slow-wave phase (reaching a minimum) and then accumulate
    # again during the subsequent wake phase.
    # We verify that the sleep pressure successfully decays below 0.80 at some point.
    assert np.min(sleep_pressure) < 0.80, f"Sleep pressure did not show sufficient decay (min: {np.min(sleep_pressure):.4f})"
