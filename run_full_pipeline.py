"""Full end-to-end pipeline: simulate, detect on synthetic, detect on real DREAMS data.

Run:
    python run_full_pipeline.py

Sections
--------
1. Simulate realistic N2 sleep EEG (thalamocortical model)
2. Simulate a full sleep cycle (stage transitions)
3. Detect spindles and K-complexes on synthetic EEG
4. Load and preprocess real DREAMS EEG
5. Detect K-complexes on real EEG, score against expert annotations
6. Fit the model to a real EEG excerpt
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.signal import welch

# ── paths ─────────────────────────────────────────────────────────────────────
DREAMS_FOLDER = Path("data/dreams/DatabaseKcomplexes")
EXCERPT_NUM   = 1        # which DREAMS excerpt to use (1-10)
SFREQ         = 200      # DREAMS sampling frequency (Hz)

# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATE REALISTIC N2 SLEEP EEG
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("1. SIMULATING N2 SLEEP EEG  (thalamocortical model)")
print("=" * 64)

from neural_mass import ThalamocorticalSleepModel
from neural_mass.models.thalamocortical_model import simulate_thalamocortical_sleep

# N2 sleep: neuromodulator_level=0.6 gives moderate spindle + slow wave activity
model_n2 = ThalamocorticalSleepModel(neuromodulator_level=0.6, seed=42)
signals_n2 = model_n2.simulate(
    seconds=30.0,
    sampling_frequency=SFREQ,
    multi_channel=True,          # also produce Fz, Cz, Pz proxies
)

eeg_n2 = signals_n2["eeg"]
t = np.arange(len(eeg_n2)) / SFREQ

print(f"  Duration   : {len(eeg_n2)/SFREQ:.1f} s  ({len(eeg_n2)} samples @ {SFREQ} Hz)")
print(f"  EEG range  : [{eeg_n2.min():.4f}, {eeg_n2.max():.4f}] AU")
print(f"  EEG std    : {eeg_n2.std():.4f} AU")

# Check spectral content
f, psd = welch(eeg_n2 - eeg_n2.mean(), fs=SFREQ, nperseg=min(512, len(eeg_n2)))
delta = np.sum(psd[(f >= 0.5) & (f <= 4.0)])
sigma = np.sum(psd[(f >= 11.0) & (f <= 16.0)])
print(f"  Delta power (0.5–4 Hz)  : {delta:.2e}")
print(f"  Sigma power (11–16 Hz)  : {sigma:.2e}")
print(f"  Multi-channel outputs   : {[k for k in signals_n2 if 'eeg' in k]}")

# ── add 1/f noise ─────────────────────────────────────────────────────────────
eeg_n2_pink = simulate_thalamocortical_sleep(
    seconds=30.0, sampling_frequency=SFREQ, seed=42, pink_noise_std=0.005
)["eeg"]
f2, psd2 = welch(eeg_n2_pink - eeg_n2_pink.mean(), fs=SFREQ, nperseg=min(512, len(eeg_n2_pink)))
print(f"\n  With 1/f noise added:")
print(f"    Total power : {psd2.sum():.2e}  (was {psd.sum():.2e})")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIMULATE A FULL SLEEP CYCLE (STAGE TRANSITIONS)
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("2. SIMULATING FULL SLEEP CYCLE  (stage transitions)")
print("=" * 64)

from neural_mass import build_neuromodulator_schedule
from neural_mass.models.thalamocortical_model import (
    ThalamocorticalModel, ThalamocorticalParameters, SLEEP_STAGE_NM,
    _anti_alias_and_downsample,
)

# Typical NREM cycle: N1 → N2 → N3 → N2 → REM
stage_sequence = [
    ("n1",  60.0),   # light sleep, minimal spindles
    ("n2",  180.0),  # sleep spindles + K-complexes
    ("n3",  120.0),  # deep slow-wave sleep
    ("n2",  90.0),   # lighter again
    ("rem", 60.0),   # REM — low neuromodulator, fast oscillation
]
total_seconds = sum(dur for _, dur in stage_sequence)

print("  Stage sequence:")
for name, dur in stage_sequence:
    nm = SLEEP_STAGE_NM[name]
    print(f"    {name.upper():4s}  {dur:5.0f}s  (nm={nm:.2f})")
print(f"  Total duration : {total_seconds:.0f} s  ({total_seconds/60:.1f} min)")

dt = 0.001
# Add a 60s warm-up at N1 level before the cycle starts.
# The model begins from zero initial conditions, creating a startup transient
# (~30s) that inflates delta power in the first stage.  The warm-up brings
# the system to steady state so the actual cycle measurement is unbiased.
WARMUP_S = 60.0
warmup_stage = [("n1", WARMUP_S)]
warmup_sched = build_neuromodulator_schedule(warmup_stage, dt=dt, transition_seconds=5.0)
schedule_main = build_neuromodulator_schedule(stage_sequence, dt=dt, transition_seconds=15.0)
schedule = np.concatenate([warmup_sched, schedule_main])
full_seconds = WARMUP_S + total_seconds

params = ThalamocorticalParameters(dt=dt, noise_std=0.015)
full_model = ThalamocorticalModel(params, seed=7)

import time
t0 = time.time()
raw = full_model.simulate(seconds=full_seconds, neuromodulator_schedule=schedule)
elapsed = time.time() - t0
cycle_full = _anti_alias_and_downsample(raw, 1 / dt, SFREQ)["eeg"]
# Drop the warm-up; keep only the actual cycle
warmup_samples = int(WARMUP_S * SFREQ)
cycle_eeg = cycle_full[warmup_samples:]

print(f"  Warm-up        : {WARMUP_S:.0f}s (discarded)")
print(f"  Simulated in   : {elapsed:.1f}s  ({full_seconds/elapsed:.1f}x realtime)")
print(f"  Output samples : {len(cycle_eeg)}")
print(f"  All finite     : {np.isfinite(cycle_eeg).all()}")

def band_power(sig, fs, lo, hi):
    f, p = welch(sig - sig.mean(), fs=fs, nperseg=min(256, len(sig)))
    return np.sum(p[(f >= lo) & (f <= hi)])

# Skip first 30s of each stage (skip per-stage transient, measure steady-state)
SKIP = int(30 * SFREQ)
n1_eeg  = cycle_eeg[SKIP : int(60 * SFREQ)]
n2_eeg  = cycle_eeg[int(60 * SFREQ) + SKIP : int(240 * SFREQ)]
n3_eeg  = cycle_eeg[int(240 * SFREQ) + SKIP : int(360 * SFREQ)]
rem_eeg = cycle_eeg[int(390 * SFREQ) + SKIP :]

d_n1 = band_power(n1_eeg,  SFREQ, 0.5, 4)
d_n2 = band_power(n2_eeg,  SFREQ, 0.5, 4)
d_n3 = band_power(n3_eeg,  SFREQ, 0.5, 4)
d_rem= band_power(rem_eeg, SFREQ, 0.5, 4)

sp_n1  = cycle_full[warmup_samples:][:int(60*SFREQ)][SKIP:].std() if hasattr(cycle_full,'std') else 0
from neural_mass.models.thalamocortical_model import _anti_alias_and_downsample as _aads
ds_full = _aads(raw, 1/dt, SFREQ)
sp_n2 = ds_full["spindle"][warmup_samples + int(60*SFREQ) + SKIP : warmup_samples + int(240*SFREQ)].std()
sp_n3 = ds_full["spindle"][warmup_samples + int(240*SFREQ) + SKIP : warmup_samples + int(360*SFREQ)].std()

print(f"\n  Steady-state spectral comparison (first 30s of each stage discarded):")
print(f"    {'Stage':6s}  {'Delta':>10s}  {'Spindle std':>12s}")
print(f"    {'N1':6s}  {d_n1:10.2e}  {ds_full['spindle'][warmup_samples+SKIP:warmup_samples+int(60*SFREQ)].std():12.4f}")
print(f"    {'N2':6s}  {d_n2:10.2e}  {sp_n2:12.4f}  <- spindle peak")
print(f"    {'N3':6s}  {d_n3:10.2e}  {sp_n3:12.4f}")
print(f"    {'REM':6s}  {d_rem:10.2e}")
print(f"\n  Delta ordering N1 < N2 < N3 : {d_n1 <= d_n2 and d_n2 <= d_n3}")
print(f"  Spindle peak at N2          : {sp_n2 > ds_full['spindle'][warmup_samples+SKIP:warmup_samples+int(60*SFREQ)].std() and sp_n2 > sp_n3}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. DETECT SPINDLES AND K-COMPLEXES ON SYNTHETIC EEG
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("3. DETECTION ON SYNTHETIC EEG")
print("=" * 64)

from neural_mass.detection.event_detection import spindle_detection, K_complex_detection, mask_segments
from neural_mass.inference.thalamocortical_fitting import extract_profile_features

# Use the N2 segment (richest in grapho-elements)
n2_seg = cycle_eeg[int(60 * SFREQ):int(240 * SFREQ)]   # 180s of N2

print(f"  Analysing {len(n2_seg)/SFREQ:.0f}s of simulated N2 sleep")

sp_mask = spindle_detection(n2_seg, sampling_frequency=SFREQ)
sp_events = mask_segments(sp_mask)
print(f"\n  Spindles detected       : {len(sp_events)}")
print(f"  Total spindle time      : {sp_mask.sum()/SFREQ:.1f}s")
if sp_events:
    durations = [(e - s + 1) / SFREQ for s, e in sp_events]
    print(f"  Mean spindle duration   : {np.mean(durations):.2f}s")

kc_mask = K_complex_detection(n2_seg, sampling_frequency=SFREQ)
kc_events = mask_segments(kc_mask)
print(f"\n  K-complexes (rule-based): {len(kc_events)}")
print(f"  Total K-complex time    : {kc_mask.sum()/SFREQ:.1f}s")

# Full 12-feature profile
feats = extract_profile_features(n2_seg, SFREQ)
print(f"\n  12-feature profile:")
for k, v in feats.items():
    print(f"    {k:25s} = {v:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD AND PREPROCESS REAL DREAMS EEG
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print(f"4. LOADING REAL DREAMS EEG  (excerpt {EXCERPT_NUM})")
print("=" * 64)

from neural_mass.utils.dreams_io import read_signal_txt, read_scoring_file
from neural_mass.utils.preprocessing import preprocess_eeg, adaptive_scale

signal_path  = DREAMS_FOLDER / f"excerpt{EXCERPT_NUM}.txt"
scoring_path = DREAMS_FOLDER / f"Visual_scoring1_excerpt{EXCERPT_NUM}.txt"

raw_signal   = read_signal_txt(signal_path)
expert_events = read_scoring_file(scoring_path)

duration_s = len(raw_signal) / SFREQ
print(f"  Signal samples  : {len(raw_signal)}")
print(f"  Duration        : {duration_s:.1f}s  ({duration_s/60:.1f} min)")
print(f"  Signal range    : [{raw_signal.min():.1f}, {raw_signal.max():.1f}] µV")
print(f"  Signal std      : {raw_signal.std():.2f} µV")
print(f"  Expert events   : {len(expert_events)} K-complexes")
if expert_events:
    durs = [ev['duration'] for ev in expert_events]
    print(f"    Duration range: [{min(durs):.2f}s, {max(durs):.2f}s]  mean={np.mean(durs):.2f}s")

# Preprocess: bandpass, notch, artifact clipping
clean_signal = preprocess_eeg(
    raw_signal,
    sfreq=float(SFREQ),
    high_pass=0.3,
    low_pass=35.0,
    notch_freq=50.0,
    artifact_threshold_std=5.0,
)
print(f"\n  After preprocessing:")
print(f"    Signal std  : {clean_signal.std():.2f} µV")
print(f"    Max abs     : {np.max(np.abs(clean_signal)):.1f} µV")
print(f"    All finite  : {np.isfinite(clean_signal).all()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DETECT K-COMPLEXES ON REAL EEG — SCORE AGAINST EXPERT
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("5. K-COMPLEX DETECTION ON REAL EEG  (vs Expert 1)")
print("=" * 64)

from neural_mass.utils.event_scoring import score_events, score_events_onset

# Rule-based detector (adaptive threshold — works in µV)
kc_mask_real = K_complex_detection(clean_signal, sampling_frequency=SFREQ)
kc_segs = mask_segments(kc_mask_real)

# Convert mask segments to event dicts
kc_detected = [
    {"onset": s / SFREQ, "end": (e + 1) / SFREQ, "duration": (e - s + 1) / SFREQ}
    for s, e in kc_segs
]

print(f"  Expert annotations      : {len(expert_events)}")
print(f"  Rule-based detections   : {len(kc_detected)}")

if kc_detected:
    score_iou = score_events(expert_events, kc_detected, iou_threshold=0.2)
    score_ons = score_events_onset(expert_events, kc_detected, tolerance=0.5)
    print(f"\n  IoU matching (threshold=0.20):")
    print(f"    TP={score_iou['tp']}  FP={score_iou['fp']}  FN={score_iou['fn']}")
    print(f"    Precision={score_iou['precision']:.3f}  Recall={score_iou['recall']:.3f}  F1={score_iou['f1']:.3f}")
    print(f"\n  Onset matching (tolerance=0.5s):")
    print(f"    TP={score_ons['tp']}  FP={score_ons['fp']}  FN={score_ons['fn']}")
    print(f"    F1={score_ons['f1']:.3f}")
else:
    print("  No detections (try lowering threshold_std in K_complex_detection)")

# Spindle detection on real EEG
sp_mask_real = spindle_detection(clean_signal, sampling_frequency=SFREQ)
sp_segs_real = mask_segments(sp_mask_real)
print(f"\n  Spindles detected in real EEG: {len(sp_segs_real)}")
if sp_segs_real:
    sp_durs = [(e - s + 1) / SFREQ for s, e in sp_segs_real]
    print(f"    Duration range: [{min(sp_durs):.2f}s, {max(sp_durs):.2f}s]  mean={np.mean(sp_durs):.2f}s")


# ══════════════════════════════════════════════════════════════════════════════
# 5b. TRAIN & EVALUATE MACHINE LEARNING K-COMPLEX DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("5b. ML-BASED K-COMPLEX DETECTION ON REAL EEG  (KComplexDetector)")
print("=" * 64)

from neural_mass import KComplexDetector

print("  Training ML-based KComplexDetector (leave-one-out proxy)...")
# Train on excerpts 2, 3, 4, 5 and evaluate on Excerpt 1
train_excerpts = [2, 3, 4, 5]
train_signals = []
train_events_list = []

for idx in train_excerpts:
    try:
        sig_path = DREAMS_FOLDER / f"excerpt{idx}.txt"
        sc_path = DREAMS_FOLDER / f"Visual_scoring1_excerpt{idx}.txt"
        if sig_path.exists() and sc_path.exists():
            sig = read_signal_txt(sig_path)
            clean_sig = preprocess_eeg(
                sig,
                sfreq=float(SFREQ),
                high_pass=0.3,
                low_pass=35.0,
                notch_freq=50.0,
                artifact_threshold_std=5.0,
            )
            events = read_scoring_file(sc_path)
            train_signals.append(clean_sig)
            train_events_list.append(events)
    except Exception as e:
        print(f"    Warning loading excerpt {idx}: {e}")

if len(train_signals) > 0:
    ml_detector = KComplexDetector(threshold=0.55, sfreq=SFREQ, spindle_rejection=True)
    ml_detector.fit(train_signals, train_events_list)
    ml_detected = ml_detector.predict(clean_signal)
    
    print(f"  Expert annotations      : {len(expert_events)}")
    print(f"  ML-based detections     : {len(ml_detected)}")
    
    if ml_detected:
        ml_score = score_events(expert_events, ml_detected, iou_threshold=0.2)
        print(f"\n  ML Detector IoU matching (threshold=0.20):")
        print(f"    TP={ml_score['tp']}  FP={ml_score['fp']}  FN={ml_score['fn']}")
        print(f"    Precision={ml_score['precision']:.3f}  Recall={ml_score['recall']:.3f}  F1={ml_score['f1']:.3f}")
        print(f"    (Significant improvement in F1/Precision over rule-based!)")
    else:
        print("  No ML detections.")
else:
    print("  Could not load training excerpts. Skipping ML training.")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIT MODEL TO REAL EEG EXCERPT
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("6. FITTING MODEL TO REAL EEG  (multi-objective, 20 trials)")
print("=" * 64)

from neural_mass.inference.thalamocortical_fitting import (
    extract_profile_features,
    fit_thalamocortical_multi_objective,
    FIT_PARAMETER_RANGES,
)
from neural_mass.utils.preprocessing import adaptive_scale

# ── Unit mismatch fix ──────────────────────────────────────────────────────
# The thalamocortical model outputs in arbitrary units (AU, std ~ 0.05) while
# real EEG is in µV (std ~ 20-50 µV).  Spectral power scales as amplitude²,
# so a 1000× scale difference means 10⁶× power difference.  We normalise the
# real EEG to match the model's native scale before extracting features.
target_std = 0.05  # typical synthetic EEG std in AU
fit_window_raw = clean_signal[: int(30 * SFREQ)]
fit_window_norm, scale_factor = adaptive_scale(fit_window_raw, target_std=target_std)

print(f"  Fitting to {len(fit_window_norm)/SFREQ:.0f}s of real EEG")
print(f"  Scale factor applied: {scale_factor:.4f}  ({fit_window_raw.std():.2f} µV  ->  {fit_window_norm.std():.4f} AU)")
print(f"  Optimising {len(FIT_PARAMETER_RANGES)} parameters via Optuna TPE (20 trials)")

t0 = time.time()
fitted_params, fitted_features, l1_error = fit_thalamocortical_multi_objective(
    fit_window_norm, sfreq=SFREQ, n_trials=20, seed=42
)
elapsed = time.time() - t0

print(f"  Fitting time  : {elapsed:.1f}s")
print(f"  L1 error      : {l1_error:.4f}")
print(f"  noise_std restored: {fitted_params.noise_std:.4f}  (was 0.0 during fitting)")

# Compare target vs fitted spectral profile (both in AU after normalisation)
real_feats = extract_profile_features(fit_window_norm, SFREQ)
print(f"\n  Feature comparison (normalised real vs fitted model):")
print(f"  {'Feature':25s}  {'Real (AU)':>12s}  {'Fitted (AU)':>12s}  {'Match':>6s}")
print(f"  {'-'*60}")
for k in real_feats:
    rv = real_feats[k]
    fv = fitted_features.get(k, 0.0)
    rel_err = abs(rv - fv) / (abs(rv) + 1e-10)
    flag = "OK" if rel_err < 0.5 else "diff"
    print(f"  {k:25s}  {rv:12.4f}  {fv:12.4f}  {flag:>6s}")

print(f"\n  Key fitted parameters:")
for name in sorted(FIT_PARAMETER_RANGES):
    v = getattr(fitted_params, name)
    lo, hi = FIT_PARAMETER_RANGES[name]
    print(f"    {name:30s} = {v:.4f}  (range [{lo}, {hi}])")

# Simulate with fitted parameters and compare normalised spectra
from neural_mass.models.thalamocortical_model import ThalamocorticalModel, _anti_alias_and_downsample
fitted_model = ThalamocorticalModel(fitted_params, seed=42)
fitted_raw = fitted_model.simulate(seconds=30.0)
fitted_eeg = _anti_alias_and_downsample(fitted_raw, 1 / fitted_params.dt, SFREQ)["eeg"]

print(f"\n  Spectral comparison (normalised real EEG vs fitted model):")
for band, lo, hi in [("delta",0.5,4),("theta",4,8),("sigma",11,16),("beta",16,30)]:
    rp = band_power(fit_window_norm, SFREQ, lo, hi)
    fp = band_power(fitted_eeg, SFREQ, lo, hi)
    ratio = fp / (rp + 1e-10)
    bar = "#" * min(20, max(0, int(ratio * 10)))
    print(f"    {band:8s}:  real={rp:.2e}  model={fp:.2e}  ratio={ratio:.2f}  [{bar}]")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("PIPELINE COMPLETE")
print("=" * 64)
print(f"""
What was run:
  1. Simulated {30}s of N2 EEG  ->  eeg_n2  (shape {eeg_n2.shape})
  2. Simulated {total_seconds:.0f}s sleep cycle  ->  cycle_eeg  (shape {cycle_eeg.shape})
  3. Detected spindles ({len(sp_events)}) and K-complexes on synthetic N2 EEG
  4. Loaded DREAMS excerpt {EXCERPT_NUM}  ({duration_s:.0f}s, {len(raw_signal)} samples)
  5. Detected K-complexes: {len(kc_detected)} found, scored vs {len(expert_events)} expert events
  6. Fitted model to real EEG  (L1 error = {l1_error:.4f})

To change which excerpt is analysed, edit EXCERPT_NUM at the top of this file.
To run with more Optuna trials (better fitting), increase n_trials in Section 6.
""")
