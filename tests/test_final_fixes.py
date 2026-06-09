import numpy as np
import time
import pytest
from scipy.signal import welch

from neural_mass import ThalamocorticalSleepModel
from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters
from neural_mass.detection.event_detection import spindle_detection, mask_segments

def test_fast_solver_speed_and_correctness():
    """Verify that the fast ODE solver executes at high speed (JIT active) and results are finite."""
    model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
    
    t0 = time.time()
    out = model.simulate(seconds=50.0, sampling_frequency=200)
    elapsed = time.time() - t0
    
    # 50.0 seconds of simulation is 50,000 steps.
    # On ordinary hardware, Numba JIT makes this execute in <0.2 seconds.
    # Even on very slow test environments, it should easily be under 1.5 seconds.
    assert elapsed < 1.5, f"Simulation was too slow: took {elapsed:.3f} seconds"
    assert np.isfinite(out["eeg"]).all()
    assert len(out["eeg"]) == 50 * 200

def test_synthetic_spindle_detection_non_zero():
    """Verify that spindle_detection successfully detects spindles in synthetic N2 EEG."""
    model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
    out = model.simulate(seconds=60.0, sampling_frequency=200)
    eeg = out["eeg"]
    
    mask = spindle_detection(eeg, sampling_frequency=200)
    events = mask_segments(mask)
    
    # With the new default spindle parameters (damping=0.90, offset=0.58) and adaptive thresholding,
    # synthetic sleep spindles should be successfully detected.
    assert len(events) > 0, "No spindles detected in N2 sleep simulation"
    print(f"Detected {len(events)} synthetic spindles.")

def test_physical_forward_model_spatial_gradients():
    """Verify that the physical forward model produces realistic fronto-parietal gradients."""
    model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=7)
    out = model.simulate(seconds=20.0, sampling_frequency=200, multi_channel=True)
    
    # Exclude early transient
    eeg_fz = out["eeg_fz"][200:]
    eeg_cz = out["eeg_cz"][200:]
    eeg_pz = out["eeg_pz"][200:]
    
    # 1. Frontal slow waves should be frontally dominant (Fz > Cz > Pz)
    # Let's check the standard deviation of the slow-wave (cortical pyramidal) state's projection.
    # Fz should have a larger cortical component standard deviation.
    std_fz = np.std(eeg_fz)
    std_pz = np.std(eeg_pz)
    
    # Since K-complexes / slow waves are frontal, Fz standard deviation should be larger than Pz.
    assert std_fz > std_pz * 1.1, f"Fz std ({std_fz:.4f}) should be larger than Pz std ({std_pz:.4f})"
    
    # 2. Spindles should be centrally maximal (Cz > Fz, Cz > Pz) and Fz/Pz should be symmetric/close.
    # Let's filter the signals in the spindle band (11-16 Hz)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [11, 16], btype="bandpass", output="sos", fs=200)
    sp_fz = sosfiltfilt(sos, eeg_fz)
    sp_cz = sosfiltfilt(sos, eeg_cz)
    sp_pz = sosfiltfilt(sos, eeg_pz)
    
    std_sp_fz = np.std(sp_fz)
    std_sp_cz = np.std(sp_cz)
    std_sp_pz = np.std(sp_pz)
    
    assert std_sp_cz > std_sp_fz * 1.1, f"Cz spindle std ({std_sp_cz:.4f}) should be larger than Fz spindle std ({std_sp_fz:.4f})"
    assert std_sp_cz > std_sp_pz * 1.1, f"Cz spindle std ({std_sp_cz:.4f}) should be larger than Pz spindle std ({std_sp_pz:.4f})"
    # Fz and Pz spindle projection should be symmetric and very close (within 5%)
    assert abs(std_sp_pz - std_sp_fz) / std_sp_fz < 0.05, f"Fz and Pz spindle stds should be close (Fz: {std_sp_fz:.4f}, Pz: {std_sp_pz:.4f})"

def test_micro_architecture_and_arousals():
    """Verify that micro-architecture and arousal modulation run without crashing."""
    model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=123)
    
    # 1. Micro-architecture
    out_ma = model.simulate(seconds=40.0, sampling_frequency=200, micro_architecture=True)
    assert np.isfinite(out_ma["eeg"]).all()
    
    # 2. Arousals
    # Arousal triggers every 90s, so we simulate 100s to ensure we hit the Wake state.
    out_ar = model.simulate(seconds=100.0, sampling_frequency=200, arousals=True)
    assert np.isfinite(out_ar["eeg"]).all()
    assert len(out_ar["eeg"]) == 100 * 200

def test_pink_noise_adds_aperiodic_spectrum():
    """Verify that pink noise adds power and correctly alters spectral features in fitting."""
    model = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=7)
    
    # Simulate with pink noise
    out_pink = model.simulate(seconds=20.0, sampling_frequency=200, pink_noise_std=0.015)
    # Simulate without pink noise
    out_clean = model.simulate(seconds=20.0, sampling_frequency=200, pink_noise_std=0.0)
    
    f, psd_pink = welch(out_pink["eeg"] - out_pink["eeg"].mean(), fs=200, nperseg=256)
    f, psd_clean = welch(out_clean["eeg"] - out_clean["eeg"].mean(), fs=200, nperseg=256)
    
    # Check overall power
    assert psd_pink.sum() > psd_clean.sum()
    
    # Check high-frequency bands (beta: 16-30 Hz) where pink noise has significant contribution
    beta_pink = np.sum(psd_pink[(f >= 16) & (f <= 30)])
    beta_clean = np.sum(psd_clean[(f >= 16) & (f <= 30)])
    assert beta_pink > beta_clean * 2.0, "Beta power should be significantly boosted by pink noise"
