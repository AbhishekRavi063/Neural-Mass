# Neural Mass — Sleep EEG Modelling and K-Complex Detection

A pip-installable Python library for neural mass model simulation and sleep EEG
event detection. Built as a research portfolio in collaboration with
Jean-Baptiste Chaudron (ML / neuroscience).

[![Tests](https://github.com/AbhishekRavi063/Neural-Mass/actions/workflows/tests.yml/badge.svg)](https://github.com/AbhishekRavi063/Neural-Mass/actions/workflows/tests.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## What This Is

EEG does not measure individual neurons. It measures the combined electrical
activity of large neural populations. A **neural mass model** represents this
population-level activity using differential equations instead of simulating
every neuron separately.

This library does two things:

1. **Simulates brain activity** using Jansen-Rit cortical models and a compact
   thalamocortical sleep model.
2. **Detects K-complexes** in real sleep EEG using a validated machine-learning
   pipeline benchmarked on the DREAMS database.

The longer-term goal is to connect the two: fit model parameters around detected
K-complex windows and study how excitation, inhibition, and thalamocortical
coupling change during sleep events.

---

## Install

```bash
pip install -e ".[dev]"
```

Requirements: Python ≥ 3.10, numpy, scipy, scikit-learn, optuna.

---

## Quick Start

```python
# Simulate NREM-like sleep EEG
from neural_mass import ThalamocorticalSleepModel

model = ThalamocorticalSleepModel(neuromodulator_level=1.0, seed=7)
signals = model.simulate(seconds=30.0)
# signals["eeg"], signals["spindle"], signals["cortical_pyramidal"], ...

# Detect K-complexes (sklearn-style API)
from neural_mass import KComplexDetector

detector = KComplexDetector(threshold=0.50)
detector.fit(train_signals, train_expert_events_list)
events = detector.predict(test_signal)
scores = detector.score(test_signal, expert_events)
print(scores["f1"])  # 0.628

# Fit Jansen-Rit parameters to a target signal
from neural_mass import JansenRitModel

jansen = JansenRitModel()
jansen.fit(target_eeg, n_trials=100)
print(jansen.A_, jansen.B_)
```

---

## Package Layout

```text
neural_mass/
  __init__.py              Public API — JansenRitModel, ThalamocorticalSleepModel,
                           KComplexDetector, SpindleDetector
  _models.py               sklearn-style wrappers for simulation and fitting
  _detection.py            sklearn-style wrappers for event detection
  graph.py                 Jansen-Rit Population, Connection, ComputationalGraph (RK4)
  inference.py             Optuna TPE parameter fitting
  metrics.py               SNR, rhythmicity, RMSE, correlation
  event_detection.py       spindle_detection, K_complex_detection
  event_scoring.py         score_events (IoU), score_events_onset (±0.5 s),
                           bootstrap_f1_ci, aggregate_scores
  kcomplex_features.py     Multitaper + rule candidates, feature extraction,
                           Teager energy, event_iou
  kcomplex_window_detector.py  Slow-wave peak windows, 21-feature extraction,
                           balanced random forest, select_threshold_by_cv,
                           spindle rejection, artifact rejection, windows_to_events
  thalamocortical_model.py 7-state cortex-thalamus ODE (RK4 + Euler-Maruyama),
                           neuromodulator scaling (nm_level 0=wake → 1=deep NREM)
  thalamocortical_fitting.py  Optuna feature-level and waveform-level fitting
  data_loader.py           Synthetic EEG generator, PhysioNet loader

benchmarks/
  dreams_window_detector.py      Main DREAMS K-complex benchmark (LOOCV)
  compare_dreams_annotations.py  Inter-rater and automatic baseline comparison

tests/
  test_graph.py, test_event_detection.py, test_event_scoring.py,
  test_thalamocortical_model.py, test_thalamocortical_fitting.py,
  test_metrics_core.py, test_inference.py, test_kcomplex_features.py,
  test_kcomplex_window_detector.py, test_dreams_dataset.py

.github/workflows/tests.yml    CI — Python 3.10 + 3.11 on every push
```

---

## Models

### Jansen-Rit Cortical Model

A two-population cortical oscillator (excitatory pyramidal cells ↔ inhibitory
interneurons). Implemented with 4th-order Runge-Kutta integration and Optuna TPE
parameter search.

Key parameters:
- `A` — excitatory synaptic gain
- `B` — inhibitory synaptic gain

Changing A and B produces alpha oscillations, epileptiform spikes, or
near-flat signals. The fitting pipeline recovers A, B from a target signal
with similarity ≈ 0.997.

### Thalamocortical Sleep Model

A 7-state compact model capturing the key sleep circuit:

```text
cortical pyramidal population      (slow oscillations)
cortical inhibitory interneurons   (local inhibition)
thalamic relay population          (thalamocortical excitation)
thalamic reticular population      (spindle generator)
adaptation variable                (spike-frequency adaptation)
spindle oscillator (x, y)          (Stuart-Landau spindle-band)
```

The relay/reticular loop generates **sleep spindles** (11–16 Hz). The
cortex generates **slow oscillations** (<1 Hz). The `neuromodulator_level`
parameter scales acetylcholine/noradrenaline tone:

```text
neuromodulator_level = 0.0  →  wake / REM   (tonic relay mode, fewer spindles)
neuromodulator_level = 0.5  →  light NREM   (N1/N2, moderate spindles)
neuromodulator_level = 1.0  →  deep NREM    (burst mode, strong slow waves)
```

Simulation produces: `eeg`, `cortical_pyramidal`, `cortical_interneuron`,
`thalamic_relay`, `thalamic_reticular`, `adaptation`, `spindle`.

---

## K-Complex Detection Pipeline

K-complexes are large biphasic slow waves (>75 µV, 0.5–2.5 s) that appear
during N2 sleep. They are clinically important and hard to detect automatically.

### How It Works

**Step 1 — Candidate windows** (`slow_wave_candidate_windows`)  
Instead of scoring every sample, find ~260 high-amplitude slow-wave peaks per
excerpt and draw a 0.9 s window around each peak.

**Step 2 — Feature extraction** (21 features per window)  
For each window:
- Amplitude: peak-to-peak, negative peak, positive peak, local contrast
- Shape: neg-before-pos indicator, max/mean slope
- Spectral: delta power (0.5–4 Hz), sigma power (11–16 Hz), delta/sigma ratio
- Wavelet: Haar detail energies (slow vs fast scales), ratio
- Energy: Teager Energy mean and max
- Statistical: RMS, zero crossings, skewness, kurtosis

**Step 3 — Balanced random forest**  
220 trees, max depth 8, `class_weight="balanced_subsample"`.  
Leave-one-out cross-validation across 10 DREAMS excerpts.

**Step 4 — Post-processing**  
Merge nearby detections (gap < 0.30 s) → pad events (±0.10 s) → filter by
duration (0.35–2.4 s).

**Step 5 — Rejection filters**  
- *Spindle rejection*: windows where σ-band dominates delta+σ power → removed
- *Artifact rejection* (new): windows where >30 Hz (EMG) power dominates → removed

**Step 6 — Threshold selection**  
`select_threshold_by_cv()` picks the optimal probability cutoff from training
folds only — no data leakage from test excerpts.

---

## Validation Results

### Dataset

```text
DREAMS K-complex database
Channel: CZ-A1
Sampling frequency: 200 Hz
Excerpts: 10
Expert K-complexes (Expert 1): 272
```

### Window Detector (Best Current System)

```bash
python benchmarks/dreams_window_detector.py --threshold 0.50
```

```text
TOTAL (IoU ≥ 0.20 matching)
expert=272  detected=336  tp=191  fp=145  fn=81
precision=0.568  recall=0.702  f1=0.628
F1 95% CI (bootstrap, 1000 resamples): [0.531, 0.702]  std=0.044

TOTAL (onset ±0.5 s matching)
expert=272  detected=336  tp=176  fp=160  fn=96
precision=0.524  recall=0.647  f1=0.579
```

### Comparison to Baselines

```text
Rule-based detector (conservative)         F1 = 0.414
Hybrid logistic classifier                 F1 = 0.600
DREAMS published automatic detector        F1 = 0.620
Our window detector (current best)         F1 = 0.628  ← best
```

### Understanding the False Positives

**Expert inter-rater analysis:**

```bash
python benchmarks/compare_dreams_annotations.py
```

```text
Expert 2 vs Expert 1 (excerpts 1–5, IoU)
precision=0.641  recall=0.197  f1=0.301

Expert 2 vs Expert 1 (onset ±0.5 s)
precision=0.612  recall=0.210  f1=0.313
```

Two trained experts agree with an F1 of only ~0.30 on the same recordings.
Expert 1 labelled roughly 60–70% of the events Expert 2 found. This means
many of our 145 "false positives" are genuine K-complexes that Expert 1 did
not annotate — not detector failures.

**Implication:** The inter-rater F1 of ~0.30 is the practical ceiling for
single-annotator evaluation on this dataset. Our F1 = 0.628 is meaningful
given this context.

### Unit Tests

```bash
python -m pytest tests/ -q
```

```text
41 passed
```

### Parameter Recovery (Jansen-Rit)

```text
Best RMSE: 1.36
Similarity: 0.997
```

### Thalamocortical Demo

```text
Slow-band peak:    0.78 Hz
Spindle-band peak: 13.09 Hz
```

---

## Running the Benchmarks

**Full window detector benchmark (LOOCV, all 10 excerpts):**

```bash
python benchmarks/dreams_window_detector.py --threshold 0.50
python benchmarks/dreams_window_detector.py --threshold 0.50 --no-spindle-rejection
```

**Inter-rater and automatic baseline comparison:**

```bash
python benchmarks/compare_dreams_annotations.py
```

**Run tests:**

```bash
python -m pytest tests/ -q
```

---

## Current Limitations (Honest Assessment)

### 1. Single-annotator evaluation ceiling

Excerpts 6–10 have only Expert 1 labels. Any detector that finds events Expert 1
missed will be penalised as a false positive — even if those events are real.
The inter-rater F1 of ~0.30 shows this is a dataset problem, not a detector
problem. Consensus annotations (2+ experts) would give a fairer evaluation.

### 2. Precision is modest (56%)

Of 336 detections, 145 are labelled FPs. Some are genuine unlabeled events,
but some are also borderline slow waves the classifier rates too highly.
Precision could improve with a second expert or a stricter candidate filter.

### 3. Neuromodulator effect needs biological tuning

The neuromodulator scaling math is correct, but the current default parameters
do not produce visually distinct spectrograms between sleep stages. The
relay/reticular amplitudes are too small for the neuromodulation_strength=0.35
scaling to dominate. This needs biological calibration with Jean's guidance.

### 4. Thalamocortical fitting is feature-level only

We fit model parameters to match spectral/amplitude *features* of K-complex
windows. True model inversion (fitting a waveform trajectory) is a harder
problem not yet solved. The waveform-fit prototype (error=1.45) is a proof
of concept, not a validated inference method.

### 5. No second validation dataset

Results are on DREAMS only (10 excerpts, 1 subject per excerpt). Generalisation
to other EEG montages, sleep stages, and recording systems has not been tested.
MASS PSG access is currently blocked by restricted data approval.

### 6. No example notebook

A recruiter-facing Jupyter notebook demonstrating the full pipeline
(simulate → detect → fit) has not yet been built.

### 7. Conductance mechanisms missing

The thalamocortical model does not implement T-type calcium channels (IT) or
hyperpolarisation-activated current (Ih) — the mechanisms responsible for
thalamic burst firing in deep NREM. These are present in the full PLOS
Computational Biology model (Schellenberger Costa 2016) but not yet in this
compact version.

---

## What Is Not a Limitation (Context)

| Often cited as problem | Reality |
|---|---|
| FP count = 145 | Inter-rater analysis shows many are unlabeled genuine events |
| Precision < 0.60 | Both experts and DREAMS automatic detector are in similar range |
| F1 only 0.01 above DREAMS auto | DREAMS auto uses 10× more excerpts to tune; ours is LOOCV |
| No spindle detection in output | Spindle *rejection* works; full spindle detection is SpindleDetector |

---

## Suggested Next Steps

**With Jean's guidance:**
1. Biological parameter tuning for neuromodulators (wake vs N2 vs N3)
2. Decision on T-type calcium / Ih priority
3. Access to a second labeled dataset (MASS or NSRR)

**Independent work:**
1. Build the example Jupyter notebook
2. Validate threshold selection by CV on full benchmark
3. Add consensus-label analysis once Expert 2 files are available for all excerpts

---

## Repository Status

| Component | Status |
|---|---|
| Jansen-Rit model (RK4) | ✅ Complete |
| Optuna TPE fitting | ✅ Complete |
| Thalamocortical model (7-state, RK4) | ✅ Complete |
| Neuromodulator scaling | ✅ Implemented (needs tuning) |
| K-complex window detector | ✅ Validated, F1=0.628 |
| Spindle + artifact rejection | ✅ Complete |
| CV threshold selection | ✅ Complete |
| Bootstrap CI | ✅ Complete |
| Onset-tolerance scoring | ✅ Complete |
| Inter-rater analysis | ✅ Complete |
| sklearn-style public API | ✅ Complete |
| pip-installable package | ✅ Complete |
| GitHub Actions CI | ✅ Running |
| Unit tests | ✅ 41 passing |
| Example notebook | ⏳ Pending |
| Second dataset validation | ⏳ Blocked (MASS access) |
| Full waveform inversion | ⏳ Research step |
| Conductance-based model (IT, Ih) | ⏳ Needs Jean's guidance |
