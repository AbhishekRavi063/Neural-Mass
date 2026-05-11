# Neural Mass EEG Modeling And K-Complex Detection

This repository is an early research prototype for building a neural mass model
library and using it as a foundation for sleep EEG analysis. The immediate goal
is to build a robust first layer: simulate EEG-like neural population activity,
fit simple model parameters, compute quality metrics, and validate K-complex
detection on a real labeled dataset.

The project intentionally skips GEDAI-style filtering for now. The current focus
is an agnostic and reproducible baseline that can later support more advanced
preprocessing, model fitting, and sleep-specific thalamocortical modeling.

## Research Motivation

EEG does not measure individual neurons. It measures the combined electrical
activity of large neural populations. A neural mass model represents this
population-level activity using differential equations instead of simulating
every neuron separately.

The biological idea is:

```text
cortical and thalamic populations interact
excitation and inhibition shape their activity
the summed activity creates oscillatory EEG-like signals
sleep events such as K-complexes appear as large slow waves
```

In this project, we currently work with two connected parts:

1. Neural mass simulation and parameter fitting.
2. K-complex detection and validation on real sleep EEG.

The longer-term goal is to connect these two parts more tightly: fit neural mass
parameters around real K-complex windows and study whether changes in excitation,
inhibition, or thalamocortical coupling can explain the observed sleep events.

## Current Model

The current simulation layer uses a Jansen-Rit-style cortical neural mass model.
It is a simplified model of interacting excitatory and inhibitory neural
populations. In the current first-layer validation, the key fitted parameters are:

```text
A = excitatory synaptic gain
B = inhibitory synaptic gain
```

Changing these parameters changes the simulated EEG-like waveform. The fitting
pipeline asks:

```text
Given a target EEG-like signal, can we recover model parameters that reproduce it?
```

Optuna is used for parameter search.

## What Has Been Implemented

### Neural Mass Core

- `Population`, `Connection`, and `ComputationalGraph` abstractions.
- Multi-population simulation with configurable coupling.
- Reproducible simulation through random seeds.
- EEG-like output generated from population state differences.

Main files:

```text
src/graph.py
src/inference.py
```

### Parameter Fitting

- Optuna-based fitting of neural mass parameters.
- Synthetic target recovery demo.
- First-layer validation script.

Main files:

```text
src/inference.py
run_optimization.py
validate_first_layer.py
```

### Metrics

The project currently reports:

```text
SNR
rhythmicity
RMSE
correlation/similarity
dominant frequency
event-level precision, recall, F1
```

Main files:

```text
src/metrics.py
src/event_scoring.py
```

### Sleep Event Detection

Implemented event detection utilities:

- spindle detection
- K-complex rule-based detection
- K-complex candidate extraction
- K-complex feature extraction
- hybrid K-complex classifier validation

The K-complex classifier uses:

```text
time-domain rule candidates
multitaper delta-energy candidates
duration and morphology features
spectral power features
Teager energy features
local contrast features
entropy, skewness, and kurtosis
```

Main files:

```text
src/event_detection.py
src/kcomplex_features.py
dreams_kcomplex_validation.py
dreams_kcomplex_classifier.py
validate_dreams_pipeline.py
```

## Dataset Used For Real Validation

The active real validation dataset is the DREAMS K-complex database:

```text
data/dreams/DatabaseKcomplexes
```

Why DREAMS is used:

- It contains real EEG excerpts.
- It includes expert K-complex event labels.
- It allows event-level scoring with TP, FP, FN, precision, recall, and F1.

Current setup:

```text
dataset: DREAMS K-complex database
channel: CZ-A1 text excerpts
sampling frequency: 200 Hz
expert labels: Visual_scoring1_excerpt*.txt
number of excerpts: 10
expert K-complexes used: 272
```

PhysioNet Sleep-EDF is not used as the main K-complex benchmark because the
current project needs event-level expert K-complex labels. PhysioNet can still
be used as an optional EEG loading experiment, but DREAMS is the real validation
path for K-complex detection.

The downloaded MASS SS2 files currently provide public expert annotations, but
the matching PSG signal files require restricted MASS access approval. Therefore
MASS is included only as a scoring scaffold for future validation.

## Current Results

### Neural Mass Parameter Recovery

Command:

```bash
python validate_first_layer.py
```

Fresh result:

```text
Best parameters: A=3.898, B=38.413
Best RMSE from optimizer: 1.3647
SNR: 9.83 dB
Rhythmicity: 0.80
Similarity: 0.9973
Dominant frequency: 5.00 Hz
Checks: PASS
```

Interpretation:

The first-layer neural mass fitting works well on a synthetic target. The model
can recover parameters close to the hidden target values and reproduce the
waveform with high similarity. This is not yet proof on real EEG, but it is a
good sanity check for the simulation and fitting layer.

### Synthetic K-Complex Test

Command:

```bash
python test_real_data.py
```

Fresh result:

```text
K-complex events injected: 2
K-complex events detected: 2
SNR: 5.05 dB
Rhythmicity: 0.37
```

Interpretation:

The detector can recover clearly injected K-complex-like waves in synthetic
EEG-like data. This is useful for debugging, but it is not enough for scientific
validation because the signal is artificial.

### DREAMS Rule Baseline

Command:

```bash
python dreams_kcomplex_validation.py --all --preset conservative
```

Fresh result:

```text
expert=272
detected=250
tp=108
fp=142
fn=164
precision=0.432
recall=0.397
f1=0.414
```

Interpretation:

The pure rule-based detector is weak. It finds some real K-complexes, but it
misses many events and produces many false positives.

### DREAMS Hybrid Classifier

Command:

```bash
python dreams_kcomplex_classifier.py --candidates hybrid --model logistic --threshold 0.75
```

Fresh result:

```text
expert=272
detected=311
tp=175
fp=136
fn=97
precision=0.563
recall=0.643
f1=0.600
```

Interpretation:

The hybrid classifier clearly improves over the rule baseline:

```text
rule baseline F1: 0.414
hybrid classifier F1: 0.600
```

The current classifier is useful, but not final. It still has many false
positives and false negatives.

### Main Validation Gate

Command:

```bash
python validate_dreams_pipeline.py
```

Fresh result:

```text
required classifier F1: 0.580
observed classifier F1: 0.600
status: PASS
```

### Unit Tests

Command:

```bash
python -m pytest
```

Fresh result:

```text
22 passed
```

## Generated Figures

Freshly generated project figures:

```text
graph_test_result.png
network_results.png
organic_comparison.png
Final_Brain_Report.png
first_layer_validation.png
real_data_test.png
kcomp_light.png
kcomp_deep.png
kcomp_messy.png
dreams_kcomplex_validation.png
dreams_kcomplex_classifier.png
```

The real benchmark figures are the DREAMS figures:

```text
dreams_kcomplex_validation.png
dreams_kcomplex_classifier.png
```

## Repository Layout

```text
src/
  graph.py              Core Population, Connection, and ComputationalGraph API
  inference.py          Optuna-based parameter fitting helpers
  metrics.py            Spectral SNR, rhythmicity, RMSE, and correlation metrics
  event_detection.py    Spindle and K-complex detection utilities
  kcomplex_features.py  Hybrid candidates and K-complex feature extraction
  event_scoring.py      Event-level scoring utilities
  data_loader.py        Synthetic EEG and optional real EEG loading helpers

run_v1_test.py                 Two-population smoke test
run_network_demo.py            Four-region network demo
run_organic_demo.py            Noise-free vs noisy simulation comparison
run_optimization.py            Parameter recovery demo
run_dashboard.py               End-to-end report generation
generate_cases.py              Synthetic K-complex case generation
validate_first_layer.py        First-layer model validation
test_real_data.py              Synthetic K-complex detection demo
dreams_kcomplex_validation.py  DREAMS rule-based detector benchmark
dreams_kcomplex_classifier.py  DREAMS hybrid classifier benchmark
validate_dreams_pipeline.py    Main DREAMS-only validation runner
research_benchmark.py          Regression gate for DREAMS classifier F1
mass_kcomplex_validation.py    MASS annotation loader/scoring scaffold
analyze_real_kcomplex.py       Optional legacy PhysioNet exploration
```

## Install

```bash
pip install -r requirements.txt
```

## Main Commands

Run tests:

```bash
python -m pytest
```

Run first-layer neural mass validation:

```bash
python validate_first_layer.py
```

Run the main DREAMS validation:

```bash
python validate_dreams_pipeline.py
```

Run the DREAMS classifier directly:

```bash
python dreams_kcomplex_classifier.py --candidates hybrid --model logistic --threshold 0.75
```

Run the benchmark gate:

```bash
python research_benchmark.py
```

Regenerate major figures:

```bash
python run_v1_test.py
python run_network_demo.py
python run_organic_demo.py
python run_dashboard.py
python validate_first_layer.py
python test_real_data.py
python generate_cases.py
python validate_dreams_pipeline.py
python dreams_kcomplex_classifier.py --candidates hybrid --model logistic --threshold 0.75
```

## Current Limitations

1. DREAMS-only validation is still small.

The current real benchmark uses 10 DREAMS excerpts and 272 expert K-complexes.
This is valid for a first benchmark, but it is not enough to claim general
performance across many subjects, channels, sleep stages, and recording systems.

2. Only one expert scorer is used so far.

The current DREAMS validation uses `Visual_scoring1_excerpt*.txt`. DREAMS also
contains second-expert labels for some excerpts. K-complex scoring is subjective,
so we need expert-vs-expert agreement before judging how close the model is to
human-level performance.

3. False positives are still high.

Current hybrid classifier:

```text
false positives: 136
precision: 0.563
```

This means many predicted K-complexes are not marked by the expert.

4. False negatives are still significant.

Current hybrid classifier:

```text
false negatives: 97
recall: 0.643
```

The detector still misses about one third of expert-marked K-complexes.

5. Candidate generation limits recall.

The classifier can only classify candidate events that were generated first. If
candidate generation misses a true K-complex, the classifier cannot recover it.

6. The multitaper path is not a full published MT-KCD reproduction.

The current multitaper candidate extraction borrows the idea of delta-band
energy enhancement, but it is not yet a complete reproduction of a published
MT-KCD algorithm.

7. Neural mass simulation and K-complex detection are not yet unified.

The neural mass model currently simulates and fits EEG-like activity. The
K-complex classifier detects events in DREAMS. The next biological step is to
fit neural mass parameters around K-complex and non-K-complex windows.

8. The current neural mass model is too simple for full sleep biology.

The current model is Jansen-Rit-style cortical modeling. K-complexes are strongly
related to thalamocortical sleep dynamics, so a future model should include a
more explicit thalamocortical sleep circuit.

9. MASS full validation is blocked by data access.

The public MASS SS2 files downloaded here include annotations, but not the
matching PSG signal files required for full event scoring. Those PSG files need
restricted access approval.

## Suggested Improvements

Short-term improvements:

1. Add DREAMS second-expert evaluation.
2. Compute expert-vs-expert agreement.
3. Compare our predictions against DREAMS automatic detection files.
4. Add onset error and duration error metrics, not only event-level F1.
5. Improve candidate generation to reduce missed events.
6. Add artifact and spindle-overlap rejection to reduce false positives.

Medium-term improvements:

1. Reproduce a published K-complex detector more closely, such as MT-KCD,
   wavelet-based detection, or Teager Energy Operator based detection.
2. Tune model thresholds using cross-validation without leaking test excerpts.
3. Try calibrated classifiers or ensembles after candidate quality improves.
4. Report performance per excerpt and analyze failure cases visually.
5. Validate on another labeled dataset if access is available.

Long-term biological improvements:

1. Add a thalamocortical neural mass model for sleep stage 2.
2. Fit model parameters around K-complex windows and non-event windows.
3. Test whether K-complexes correspond to changes in excitation, inhibition, or
   thalamocortical coupling.
4. Build a pipeline that explains detected sleep events through fitted model
   parameters, not only event classification.

## Questions For Guide Review

1. Is DREAMS acceptable as the first real validation dataset, or should we seek
   access to MASS PSG before extending the detector further?

2. Should the next milestone focus on improving K-complex detection accuracy, or
   on connecting detected events to neural mass parameter fitting?

3. Which biological model should be prioritized next: a stronger cortical
   Jansen-Rit model or an explicit thalamocortical sleep model?

4. What event matching criterion should be used for reporting: IoU overlap,
   onset tolerance, duration tolerance, or multiple metrics together?

5. Should second-expert agreement be treated as the next validation baseline
   before trying to claim model performance?

6. Is the current F1 target of 0.58 acceptable for a prototype gate, or should a
   higher benchmark be set before the next phase?

7. Which published K-complex detection method should be reproduced first for
   comparison: multitaper/MT-KCD, wavelet/TQWT, or Teager Energy Operator?

## Status Summary

The project currently has a working neural mass simulation layer, parameter
fitting, metrics, synthetic K-complex tests, and DREAMS-based real K-complex
validation. The DREAMS hybrid classifier improves clearly over the rule baseline,
but the project is still a prototype. The next major research step is to compare
against multiple human/automatic DREAMS labels and then connect K-complex windows
back to neural mass model parameters.
