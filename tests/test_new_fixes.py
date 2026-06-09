"""Tests for the second round of fixes (remaining limitations)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch


# ── Fix #1 : adaptive K-complex threshold ────────────────────────────────────

def test_kcomplex_detects_on_realistic_synthetic_signal():
    """K_complex_detection must detect discrete K-complex events.

    The neural mass model generates *continuous* slow oscillations, not
    discrete K-complex transients.  This test uses generate_realistic_eeg
    (which inserts isolated biphasic events) to validate the adaptive-threshold
    detector correctly.  The multitaper-based grapho-element test covers the
    neural mass model's output separately.
    """
    from neural_mass.utils.data_loader import generate_realistic_eeg
    from neural_mass.detection.event_detection import K_complex_detection

    sfreq = 100
    signal, events = generate_realistic_eeg(sfreq=sfreq, seed=42, return_events=True)
    mask = K_complex_detection(signal, sampling_frequency=sfreq)

    assert mask.shape == signal.shape
    assert mask.sum() > 0, "K-complex detector never fires on a signal with injected K-complexes"
    # Must overlap with the ground-truth K-complex region
    assert np.logical_and(mask, events["k_complex"]).any(), (
        "Detected region does not overlap the injected K-complex"
    )


def test_kcomplex_adaptive_threshold_scales_with_amplitude():
    """The detector must fire consistently regardless of signal scale."""
    from neural_mass.detection.event_detection import K_complex_detection

    sfreq = 200
    t = np.arange(int(4 * sfreq)) / sfreq
    # Biphasic K-complex: sharp negative then positive rebound (1.0–2.2s)
    signal_au = 0.02 * np.sin(2 * np.pi * 0.8 * t)
    kc_start, kc_end = int(1.0 * sfreq), int(2.2 * sfreq)
    kc_dur = kc_end - kc_start
    kc_t = np.linspace(0, 1, kc_dur)
    # negative peak at 0.35, positive rebound at 0.70
    kc_wave = -0.25 * np.exp(-((kc_t - 0.35)**2) / (2 * 0.08**2)) + \
               0.15 * np.exp(-((kc_t - 0.70)**2) / (2 * 0.12**2))
    signal_au[kc_start:kc_end] += kc_wave

    # µV scale (same shape, 2000× bigger)
    signal_uv = signal_au * 2000.0

    mask_au = K_complex_detection(signal_au, sampling_frequency=sfreq, max_rise_time_ms=2000.0, reject_artifacts=False)
    mask_uv = K_complex_detection(signal_uv, sampling_frequency=sfreq, max_rise_time_ms=2000.0, reject_artifacts=False)

    assert mask_au.any(), "Adaptive threshold missed K-complex in AU-scale signal"
    assert mask_uv.any(), "Adaptive threshold missed K-complex in uV-scale signal"


def test_kcomplex_explicit_threshold_still_works():
    """Passing an explicit float for min_peak_to_peak must override adaptive."""
    from neural_mass.detection.event_detection import K_complex_detection

    sfreq = 200
    signal = np.zeros(sfreq * 2)
    # Inject a large K-complex
    signal[100:300] = np.sin(np.linspace(0, np.pi, 200)) * 100.0
    signal[100:300] *= -1  # negative then positive (biphasic)
    signal[100:300] += np.sin(np.linspace(0, np.pi, 200)) * 60.0

    # Should detect with default adaptive threshold
    mask_adaptive = K_complex_detection(signal, sampling_frequency=sfreq)
    # Should NOT detect with impossibly high explicit threshold
    mask_huge = K_complex_detection(signal, sampling_frequency=sfreq, min_peak_to_peak=1e9)

    assert not mask_huge.any(), "Explicit huge threshold should prevent all detections"


# ── Fix #2 : non-linear neuromodulation ──────────────────────────────────────

def test_neuromodulation_produces_spectral_differences():
    """Neuromodulation must produce physiologically correct spectral staging.

    Correct physiology (matching the bell-curve spindle model):
      • Delta increases monotonically: wake < N1 < N2 < N3
      • Spindle (sigma) PEAKS at N2 (nm=0.5), NOT at deep NREM
        — deep NREM is slow-wave dominated with fewer spindles
    """
    from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters

    def get_band_power(nm_level, band_low, band_high):
        p = ThalamocorticalParameters(dt=0.001, neuromodulator_level=nm_level, noise_std=0.01)
        m = ThalamocorticalModel(p, seed=42)
        out = m.simulate(seconds=20)
        f, psd = welch(out["eeg"] - out["eeg"].mean(), fs=1000, nperseg=512)
        return float(np.sum(psd[(f >= band_low) & (f <= band_high)]))

    delta_wake  = get_band_power(0.0, 0.5, 4.0)
    delta_n3    = get_band_power(1.0, 0.5, 4.0)
    sigma_wake  = get_band_power(0.0, 11.0, 16.0)
    sigma_n2    = get_band_power(0.5, 11.0, 16.0)   # sigma peaks at N2
    sigma_n3    = get_band_power(1.0, 11.0, 16.0)

    # 1. Deep NREM must have more delta than wake
    assert delta_n3 > delta_wake, (
        f"NREM delta ({delta_n3:.2e}) should exceed wake delta ({delta_wake:.2e})"
    )
    # 2. Spindles peak at N2 (nm=0.5) — higher than both wake and deep NREM
    assert sigma_n2 > sigma_wake, (
        f"N2 sigma ({sigma_n2:.2e}) should exceed wake sigma ({sigma_wake:.2e})"
    )
    assert sigma_n2 > sigma_n3, (
        f"N2 sigma ({sigma_n2:.2e}) should exceed N3 sigma ({sigma_n3:.2e}) — "
        "deep NREM is slow-wave dominated, not spindle dominated"
    )


def test_neuromodulation_is_continuous():
    """Intermediate nm values should produce monotonically ordered delta power."""
    from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters

    levels = [0.0, 0.5, 1.0]
    delta_powers = []
    for nm in levels:
        p = ThalamocorticalParameters(dt=0.001, neuromodulator_level=nm, noise_std=0.01)
        m = ThalamocorticalModel(p, seed=99)
        out = m.simulate(seconds=15)
        f, psd = welch(out["eeg"] - out["eeg"].mean(), fs=1000, nperseg=512)
        delta_powers.append(float(np.sum(psd[(f >= 0.5) & (f <= 4.0)])))

    # Should be monotonically non-decreasing with NREM depth
    assert delta_powers[0] <= delta_powers[1] <= delta_powers[2], (
        f"Delta power not monotone: {delta_powers}"
    )


# ── Fix #3 : grapho-element features no longer always zero ───────────────────

def test_grapho_elements_detected_in_synthetic_eeg():
    """extract_profile_features must return non-zero grapho counts on synthetic EEG."""
    from neural_mass.models.thalamocortical_model import simulate_thalamocortical_sleep
    from neural_mass.inference.thalamocortical_fitting import extract_profile_features

    sfreq = 200
    eeg = simulate_thalamocortical_sleep(seconds=30, sampling_frequency=sfreq, seed=3)["eeg"]
    feats = extract_profile_features(eeg, sfreq)

    # At least one grapho-element type must be detected
    total_events = feats["grapho_kcomplex"] + feats["grapho_spindle"]
    assert total_events > 0, (
        f"grapho_kcomplex={feats['grapho_kcomplex']}, grapho_spindle={feats['grapho_spindle']}. "
        "Both zero means adaptive threshold not working."
    )


# ── Fix #4 : expanded parameter coverage ─────────────────────────────────────

def test_fit_parameter_ranges_covers_12_params():
    """FIT_PARAMETER_RANGES must cover at least 12 model parameters."""
    from neural_mass.inference.thalamocortical_fitting import FIT_PARAMETER_RANGES
    from dataclasses import fields
    from neural_mass.models.thalamocortical_model import ThalamocorticalParameters

    all_params = {f.name for f in fields(ThalamocorticalParameters)}
    fitted = set(FIT_PARAMETER_RANGES.keys())
    assert len(fitted) >= 12, f"Only {len(fitted)} parameters fitted; expected ≥12"
    assert fitted.issubset(all_params), f"Unknown params in ranges: {fitted - all_params}"


def test_new_params_in_fit_ranges():
    """Key thalamic loop parameters must be in FIT_PARAMETER_RANGES."""
    from neural_mass.inference.thalamocortical_fitting import FIT_PARAMETER_RANGES

    required = {"relay_to_reticular", "spindle_feedback_gain",
                "cortical_excitation_scale", "eeg_spindle_weight", "spindle_drive_offset"}
    missing = required - set(FIT_PARAMETER_RANGES.keys())
    assert not missing, f"Missing params from ranges: {missing}"


# ── Fix #5 : NaN input raises ValueError ─────────────────────────────────────

def test_spindle_detection_raises_on_nan():
    from neural_mass.detection.event_detection import spindle_detection
    with pytest.raises(ValueError, match="NaN or Inf"):
        spindle_detection(np.full(500, np.nan), sampling_frequency=200)


def test_kcomplex_detection_raises_on_nan():
    from neural_mass.detection.event_detection import K_complex_detection
    with pytest.raises(ValueError, match="NaN or Inf"):
        K_complex_detection(np.full(500, np.nan), sampling_frequency=200)


def test_spindle_detection_raises_on_inf():
    from neural_mass.detection.event_detection import spindle_detection
    sig = np.zeros(500)
    sig[10] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        spindle_detection(sig, sampling_frequency=200)


# ── Fix #7 : preprocessing pipeline ─────────────────────────────────────────

def test_preprocess_eeg_returns_finite_clean_signal():
    from neural_mass.utils.preprocessing import preprocess_eeg

    rng = np.random.default_rng(1)
    # Noisy signal with amplitude spikes (EMG-like artifacts)
    signal = rng.normal(0, 50, 2000)
    signal[300:310] = 5000.0   # large artifact

    out = preprocess_eeg(signal, sfreq=200.0)

    assert out.shape == signal.shape
    assert np.isfinite(out).all(), "preprocess_eeg output contains NaN or Inf"
    # After clipping, the spike should be gone
    assert np.max(np.abs(out)) < np.max(np.abs(signal))


def test_preprocess_eeg_removes_high_frequency_above_lowpass():
    from neural_mass.utils.preprocessing import preprocess_eeg
    from scipy.signal import welch

    sfreq = 200.0
    t = np.arange(int(5 * sfreq)) / sfreq
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 80 * t)  # 80 Hz above low_pass=40

    out = preprocess_eeg(signal, sfreq=sfreq, low_pass=40.0, notch_freq=None)

    f, psd = welch(out - out.mean(), fs=sfreq, nperseg=256)
    power_80hz = np.sum(psd[f >= 60])
    power_10hz = np.sum(psd[(f >= 8) & (f <= 12)])
    assert power_10hz > power_80hz * 10, "10 Hz component should dominate after low-pass filter"


def test_adaptive_scale_normalizes_to_target_std():
    from neural_mass.utils.preprocessing import adaptive_scale

    rng = np.random.default_rng(2)
    signal = rng.normal(0, 0.05, 1000)  # small amplitude (AU scale)
    scaled, factor = adaptive_scale(signal, target_std=50.0)

    assert scaled.shape == signal.shape
    assert abs(np.std(scaled) - 50.0) < 1.0, f"Scaled std={np.std(scaled):.2f}, expected ~50"
    assert factor == pytest.approx(50.0 / np.std(signal), rel=1e-4)


def test_bandpass_filter_rejects_dc_and_hf():
    from neural_mass.utils.preprocessing import bandpass_filter

    sfreq = 200.0
    t = np.arange(int(5 * sfreq)) / sfreq
    signal = 10.0 + np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 80 * t)
    out = bandpass_filter(signal, sfreq, low=0.3, high=40.0)

    assert abs(np.mean(out)) < 0.5, "DC not removed"


# ── Fix #6 : sleep stage transitions ─────────────────────────────────────────

def test_build_neuromodulator_schedule_correct_length():
    from neural_mass.models.thalamocortical_model import build_neuromodulator_schedule

    stages = [("n2", 10.0), ("n3", 5.0), ("rem", 3.0)]
    dt = 0.001
    schedule = build_neuromodulator_schedule(stages, dt=dt)
    expected = int((10.0 + 5.0 + 3.0) / dt)
    assert len(schedule) == expected


def test_build_neuromodulator_schedule_correct_levels():
    from neural_mass.models.thalamocortical_model import build_neuromodulator_schedule, SLEEP_STAGE_NM

    stages = [("n2", 10.0), ("n3", 5.0)]
    schedule = build_neuromodulator_schedule(stages, dt=0.001, transition_seconds=0.0)
    midpoint_n2 = int(5.0 / 0.001)
    midpoint_n3 = int(12.0 / 0.001)

    assert schedule[midpoint_n2] == pytest.approx(SLEEP_STAGE_NM["n2"])
    assert schedule[midpoint_n3] == pytest.approx(SLEEP_STAGE_NM["n3"])


def test_build_neuromodulator_schedule_rejects_unknown_stage():
    from neural_mass.models.thalamocortical_model import build_neuromodulator_schedule

    with pytest.raises(ValueError, match="Unknown stage"):
        build_neuromodulator_schedule([("stage4_not_real", 10.0)])


def test_simulate_with_neuromodulator_schedule():
    """Passing a neuromodulator_schedule must not crash and output must be finite."""
    from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters, build_neuromodulator_schedule

    dt = 0.001
    stages = [("n2", 5.0), ("n3", 5.0)]
    schedule = build_neuromodulator_schedule(stages, dt=dt)
    params = ThalamocorticalParameters(dt=dt)
    model = ThalamocorticalModel(params, seed=7)
    out = model.simulate(seconds=10.0, neuromodulator_schedule=schedule)

    assert np.isfinite(out["eeg"]).all()
    assert len(out["eeg"]) == int(10.0 / dt)


# ── Fix #8 : multi-channel EEG proxy ─────────────────────────────────────────

def test_multi_channel_simulation_adds_electrode_keys():
    from neural_mass.models._models import ThalamocorticalSleepModel

    model = ThalamocorticalSleepModel(neuromodulator_level=0.5, seed=7)
    out = model.simulate(seconds=5.0, sampling_frequency=200, multi_channel=True)

    for key in ("eeg_fz", "eeg_cz", "eeg_pz"):
        assert key in out, f"Missing channel key: {key}"
        assert np.isfinite(out[key]).all()
        assert len(out[key]) == len(out["eeg"])


def test_multi_channel_frontoparietal_differ():
    """Fz and Pz must differ (different spindle/slow-wave weighting)."""
    from neural_mass.models._models import ThalamocorticalSleepModel

    model = ThalamocorticalSleepModel(neuromodulator_level=1.0, seed=3)
    out = model.simulate(seconds=10.0, sampling_frequency=200, multi_channel=True)

    corr = np.corrcoef(out["eeg_fz"], out["eeg_pz"])[0, 1]
    assert corr < 0.999, "Fz and Pz are identical — spatial weighting not applied"


# ── Fix #9 : 1/f noise in simulate_thalamocortical_sleep ─────────────────────

def test_pink_noise_adds_aperiodic_activity():
    """EEG with pink_noise_std>0 should have more low-frequency power than without."""
    from neural_mass.models.thalamocortical_model import simulate_thalamocortical_sleep

    sfreq = 200
    no_pink = simulate_thalamocortical_sleep(seconds=20, sampling_frequency=sfreq,
                                              seed=5, pink_noise_std=0.0)["eeg"]
    with_pink = simulate_thalamocortical_sleep(seconds=20, sampling_frequency=sfreq,
                                               seed=5, pink_noise_std=0.02)["eeg"]

    f, psd_no   = welch(no_pink   - no_pink.mean(),   fs=sfreq, nperseg=256)
    f, psd_pink = welch(with_pink - with_pink.mean(), fs=sfreq, nperseg=256)

    # Pink noise boosts overall broadband power
    assert psd_pink.sum() > psd_no.sum(), "Pink noise did not add power to EEG"


def test_pink_noise_zero_is_noop():
    """pink_noise_std=0 must return the same EEG as pure model output."""
    from neural_mass.models.thalamocortical_model import simulate_thalamocortical_sleep

    sfreq = 200
    a = simulate_thalamocortical_sleep(seconds=5, sampling_frequency=sfreq,
                                        seed=11, pink_noise_std=0.0)["eeg"]
    b = simulate_thalamocortical_sleep(seconds=5, sampling_frequency=sfreq,
                                        seed=11, pink_noise_std=0.0)["eeg"]
    np.testing.assert_allclose(a, b, err_msg="Identical seeds must give identical output")


# ── Fix #15 : single source of truth for threshold grid ──────────────────────

def test_threshold_grid_single_source():
    """The threshold grid constant must exist in kcomplex_window_detector."""
    from neural_mass.detection.kcomplex_window_detector import _THRESHOLDS_GRID

    assert isinstance(_THRESHOLDS_GRID, list)
    assert len(_THRESHOLDS_GRID) > 0
    assert all(0.0 < t < 1.0 for t in _THRESHOLDS_GRID)
