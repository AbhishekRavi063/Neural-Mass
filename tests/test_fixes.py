"""Regression tests covering every fix applied in the audit."""
from __future__ import annotations

import numpy as np
import pytest

# ── Fix #1 : noise only on velocity states ────────────────────────────────────

def test_noise_does_not_corrupt_position_states():
    """Position states must evolve via velocity, not jump from direct noise."""
    from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters

    # Use a HIGH noise model so differences are obvious
    params = ThalamocorticalParameters(noise_std=1.0, dt=0.001)
    rng = np.random.default_rng(0)

    # Collect 200 steps and look at per-step deltas for position states
    model = ThalamocorticalModel(params, seed=0)
    prev = model.state.copy()
    position_jumps = []
    for _ in range(200):
        model.step()
        # Position state 0 (cortical pyramidal) — its change should come from velocity
        # (state 1), not from direct noise. If noise were injected there, the jump
        # would be O(noise_std * sqrt(dt)) ≈ 0.032, far larger than dt * velocity.
        jump = abs(model.state[0] - prev[0])
        position_jumps.append(jump)
        prev = model.state.copy()

    # With dt=0.001 and velocity ~ O(1) the expected position change ≈ 0.001.
    # Direct noise injection at noise_std=1 would produce jumps ~ 0.032.
    # Threshold of 0.02 separates the two regimes.
    assert np.mean(position_jumps) < 0.02, (
        f"Position state receiving direct noise: mean jump = {np.mean(position_jumps):.4f}"
    )


# ── Fix #6 : anti-aliasing before downsampling ───────────────────────────────

def test_downsampling_preserves_slow_wave_and_suppresses_alias():
    """Low-pass filter before stride keeps slow content and removes alias risk."""
    from neural_mass.models.thalamocortical_model import simulate_thalamocortical_sleep
    from scipy.signal import welch

    sfreq = 200
    signals = simulate_thalamocortical_sleep(seconds=10, sampling_frequency=sfreq, seed=5)
    eeg = signals["eeg"]

    freqs, psd = welch(eeg - np.mean(eeg), fs=sfreq, nperseg=min(512, len(eeg)))
    # Slow-wave content must survive the downsampling
    slow_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4.0)])
    # Signal above Nyquist must be absent (anti-alias filter removes it)
    above_nyq = np.sum(psd[freqs >= sfreq / 2 - 1])

    assert slow_power > 0, "Slow-wave power lost after anti-aliasing filter"
    assert above_nyq < 1e-6, f"Energy above Nyquist present (alias artefact): {above_nyq}"


# ── Fix #10 : fit() must not overwrite self.A/self.B ─────────────────────────

def test_jansenrit_fit_does_not_overwrite_constructor_params():
    """self.A and self.B must retain their original values after fit()."""
    from neural_mass.models.graph import ComputationalGraph, Connection, Population

    from neural_mass.models._models import JansenRitModel

    original_A, original_B = 3.25, 22.0
    model = JansenRitModel(A=original_A, B=original_B)

    # Generate a trivial target
    cortex = Population(A=4.0, B=30.0)
    thal = Population(A=4.0, B=30.0)
    graph = ComputationalGraph([cortex, thal], [Connection(cortex, thal, 10), Connection(thal, cortex, 10)], seed=1)
    target = graph.simulate(steps=200)[:, 0]

    model.fit(target, n_trials=3)

    assert model.A == pytest.approx(original_A), "self.A was mutated during fit()"
    assert model.B == pytest.approx(original_B), "self.B was mutated during fit()"
    assert model.A_ is not None, "self.A_ not set after fit()"
    assert model.B_ is not None, "self.B_ not set after fit()"


def test_jansenrit_simulate_uses_fitted_params():
    """simulate() after fit() must use A_ / B_, not the original A / B."""
    from neural_mass.models._models import JansenRitModel

    model = JansenRitModel(A=3.25, B=22.0, seed=1)
    # Simulate before fit
    before = model.simulate(seconds=0.5)

    # Manually set fitted params to something very different
    model.A_ = 8.0
    model.B_ = 50.0
    after = model.simulate(seconds=0.5)

    # They must differ because simulate() picks up A_/B_
    assert not np.allclose(before, after), "simulate() ignores fitted parameters"


# ── Fix #11 : SpindleDetector.predict_events() ───────────────────────────────

def test_spindle_detector_predict_events_returns_dicts():
    """predict_events() must return a list of dicts with onset/end/duration keys."""
    from neural_mass.detection._detection import SpindleDetector

    sfreq = 200.0
    t = np.arange(int(5 * sfreq)) / sfreq
    signal = 0.05 * np.sin(2 * np.pi * 2 * t)
    burst = (t >= 2.0) & (t < 3.0)
    signal[burst] += 4.0 * np.sin(2 * np.pi * 13.0 * t[burst])

    detector = SpindleDetector(sfreq=sfreq)
    events = detector.predict_events(signal)

    assert isinstance(events, list)
    assert len(events) > 0
    for ev in events:
        assert "onset" in ev
        assert "end" in ev
        assert "duration" in ev
        assert ev["duration"] > 0
        assert ev["end"] > ev["onset"]


def test_spindle_detector_predict_events_consistent_with_mask():
    """predict_events() events must correspond to the boolean mask from predict()."""
    from neural_mass.detection._detection import SpindleDetector

    sfreq = 200.0
    t = np.arange(int(5 * sfreq)) / sfreq
    signal = 0.05 * np.sin(2 * np.pi * 2 * t)
    signal[(t >= 2.0) & (t < 3.0)] += 4.0 * np.sin(2 * np.pi * 13.0 * t[(t >= 2.0) & (t < 3.0)])

    detector = SpindleDetector(sfreq=sfreq)
    mask = detector.predict(signal)
    events = detector.predict_events(signal)

    # Both should agree on whether spindles were detected
    assert bool(mask.any()) == (len(events) > 0)


# ── Fix #14 : module-level import in kcomplex_window_detector ─────────────────

def test_kcomplex_window_detector_imports_cleanly():
    """Importing kcomplex_window_detector must not raise ImportError."""
    import importlib
    mod = importlib.import_module("neural_mass.detection.kcomplex_window_detector")
    assert hasattr(mod, "score_events"), "score_events must be importable at module level"


# ── Fix #31 : SNR must not return infinity ────────────────────────────────────

def test_snr_returns_finite_for_pure_tone():
    """A pure sinusoid (zero noise band) must not return inf."""
    from neural_mass.utils.metrics import calculate_snr

    sfreq = 200
    t = np.arange(1000) / sfreq
    pure_tone = np.sin(2 * np.pi * 10 * t)

    snr = calculate_snr(pure_tone, sfreq)

    assert np.isfinite(snr), f"SNR returned non-finite value: {snr}"
    assert snr > 0


def test_snr_finite_for_flat_spectrum():
    """White noise has no dominant peak; SNR must still be finite."""
    from neural_mass.utils.metrics import calculate_snr

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 1000)
    snr = calculate_snr(noise, sampling_frequency=200)
    assert np.isfinite(snr)


# ── Fix #27 : dreams_io robust parsing ────────────────────────────────────────

def test_scoring_file_skips_malformed_lines(tmp_path):
    """Malformed lines (non-numeric, wrong columns) must be silently skipped."""
    from neural_mass.utils.dreams_io import read_scoring_file

    content = "\n".join([
        "[Header line]",
        "1.5 0.8",          # valid
        "bad_onset 0.5",    # non-numeric onset → skip
        "3.0 garbage",      # non-numeric duration → skip
        "5.0",              # only one column → skip
        "7.0 1.2",          # valid
        "",                 # blank → skip
    ])
    f = tmp_path / "scoring.txt"
    f.write_text(content)

    events = read_scoring_file(f)

    assert len(events) == 2
    assert events[0]["onset"] == pytest.approx(1.5)
    assert events[1]["onset"] == pytest.approx(7.0)


def test_scoring_file_skips_non_positive_duration(tmp_path):
    """Events with duration ≤ 0 must be dropped."""
    from neural_mass.utils.dreams_io import read_scoring_file

    content = "2.0 0.0\n4.0 -0.5\n6.0 1.0\n"
    f = tmp_path / "scoring.txt"
    f.write_text(content)

    events = read_scoring_file(f)

    assert len(events) == 1
    assert events[0]["onset"] == pytest.approx(6.0)


def test_signal_file_skips_malformed_lines(tmp_path):
    """Non-numeric sample lines must be silently skipped."""
    from neural_mass.utils.dreams_io import read_signal_txt

    content = "[Header]\n1.0\n2.0\nNaN_text\n3.0\n"
    f = tmp_path / "signal.txt"
    f.write_text(content)

    arr = read_signal_txt(f)

    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


# ── Noise_std restored after fitting ─────────────────────────────────────────

def test_fitted_params_restore_noise_std():
    """Returned parameters from fitting must have noise_std > 0 (not locked at 0)."""
    from neural_mass.inference.thalamocortical_fitting import fit_thalamocortical_features, extract_window_features
    from neural_mass.models.thalamocortical_model import ThalamocorticalParameters

    sfreq = 200
    t = np.arange(1000) / sfreq
    signal = np.sin(2 * np.pi * 0.8 * t)
    target = extract_window_features(signal, sfreq)

    params, _, _ = fit_thalamocortical_features(target, seconds=2.0, sfreq=sfreq, n_trials=2, seed=1)

    default_noise = ThalamocorticalParameters().noise_std
    assert params.noise_std == pytest.approx(default_noise), (
        f"noise_std stuck at 0; expected {default_noise}, got {params.noise_std}"
    )


# ── Spindle filter order robustness ──────────────────────────────────────────

def test_spindle_detection_on_short_signal_does_not_crash():
    """Short signals (< 1 s) must not raise from aggressive filter order."""
    from neural_mass.detection.event_detection import spindle_detection

    sfreq = 200
    # 0.5 s signal — order 10 butter + sosfiltfilt would fail here
    signal = np.random.default_rng(1).normal(0, 1, sfreq // 2)
    result = spindle_detection(signal, sampling_frequency=sfreq)
    assert result.shape == signal.shape
    assert result.dtype == bool
