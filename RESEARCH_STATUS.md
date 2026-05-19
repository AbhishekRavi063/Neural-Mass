# Research Status

This project is still a research prototype, but it now has a reproducible
validation path for K-complex detection and basic tests for the neural mass
simulation layer. It now also includes a first compact thalamocortical neural
mass model for NREM-like slow and spindle-band rhythms.

## Current Validated Dataset

DREAMS K-complex database:

- 10 EEG excerpts
- channel used: `CZ-A1`
- sampling frequency: 200 Hz
- expert labels: `Visual_scoring1_excerpt*.txt`
- total expert K-complexes in current validation: 272

PhysioNet is not used as the main K-complex benchmark in this project. It can
still be used for exploratory EEG loading, but the active event-level validation
path is DREAMS because DREAMS includes accessible expert K-complex labels.

## Current Best DREAMS Result

Workflow:

1. Hybrid loose K-complex candidate generation:
   - time-domain rule candidates
   - multitaper delta-energy candidates
2. Multi-domain feature extraction:
   - duration
   - peak-to-peak amplitude
   - negative and positive peak features
   - local amplitude contrast
   - slopes
   - spectral powers
   - Teager energy features
   - entropy, skewness, kurtosis
3. Leave-one-excerpt-out classifier validation.
4. Micro-averaged event scoring using IoU matching.

Current default command:

```bash
python validate_dreams_pipeline.py
```

Current result:

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

## Best Current Detector

The best current detector is the window-based wavelet/TEO detector:

```bash
python dreams_window_detector.py --threshold 0.50
```

It uses slow-wave peak candidate windows, 0.3-30 Hz preprocessing,
wavelet/Haar energy features, Teager Energy Operator features, delta/sigma
spectral features, morphology features, a balanced random forest, and event
post-processing.

Current result:

```text
expert=272
detected=336
tp=191
fp=145
fn=81
precision=0.568
recall=0.702
f1=0.628
```

Comparison:

```text
old hybrid classifier F1: 0.600
DREAMS automatic detector F1: 0.620
new window detector F1: 0.628
```

## Current Thalamocortical Model Status

The new thalamocortical layer is implemented in:

```text
src/thalamocortical_model.py
src/thalamocortical_fitting.py
run_thalamocortical_demo.py
analyze_dreams_thalamocortical_fit.py
fit_dreams_kcomplex_waveform.py
```

The model includes:

```text
cortical pyramidal population
cortical inhibitory interneuron population
thalamic relay population
thalamic reticular population
cortical adaptation
spindle-band thalamic oscillator
cortex-thalamus coupling
```

Fresh demo result:

```text
slow-band peak: 0.78 Hz
spindle-band peak: 13.09 Hz
```

This is the first implementation step toward Jean's requested thalamus extension.
It is not yet a full reproduction of Schellenberger Costa et al. 2016, but it
gives the project a testable cortex-thalamus structure for the next fitting
experiments.

## DREAMS K-Complex Vs Control Model-Fit Result

The current integration analysis compares 60 expert DREAMS K-complex windows with
60 matched non-event control windows. It extracts EEG features and fits the
compact thalamocortical model separately to each condition.

Command:

```bash
python analyze_dreams_thalamocortical_fit.py
```

Current feature contrast:

```text
K-complex peak_to_peak: 121.25
Control peak_to_peak: 80.06
K-complex slow_power: 3059.10
Control slow_power: 586.41
K-complex slow_to_spindle_ratio: 57.28
Control slow_to_spindle_ratio: 24.46
```

Current compact fitted parameter contrast, K-complex minus control:

```text
adaptation_strength: -0.0929
cortex_to_thalamus: +0.4455
thalamus_to_cortex: +0.0502
reticular_inhibition: +0.4388
background_drive: +0.4767
cortical_damping: -0.0010
spindle_damping: +0.2103
```

Interpretation: in this compact model, K-complex windows are associated with
larger slow-wave amplitude/power and a fitted shift toward stronger
cortex-to-thalamus drive, stronger reticular inhibition, and higher background
drive. This should be treated as an early hypothesis because the fit is currently
feature-level, not full waveform-level inversion.

## Human And Automatic DREAMS Baselines

The project now compares available DREAMS annotations and automatic detections:

```text
Expert2 vs Expert1, excerpts 1-5:
precision=0.641
recall=0.197
f1=0.301

DREAMS automatic vs Expert1, excerpts 1-10:
precision=0.622
recall=0.618
f1=0.620

Old hybrid classifier vs Expert1, excerpts 1-10:
precision=0.563
recall=0.643
f1=0.600

New window detector vs Expert1, excerpts 1-10:
precision=0.568
recall=0.702
f1=0.628
```

The new window detector is slightly ahead of the DREAMS automatic detector. The
second-expert result is low under the current IoU matching rule, so K-complex
agreement should be evaluated under multiple matching rules before making strong
claims.

## Waveform-Level Fit Prototype

The project now includes an average-waveform fit:

```bash
python fit_dreams_kcomplex_waveform.py
```

Current result:

```text
K-complex windows used: 40
Control windows used: 40
Waveform fit error: 1.44794
```

This is the first waveform-level bridge between DREAMS K-complex morphology and
the compact thalamocortical model. It is still a random-search prototype and not
a final inversion method.

## Regression Gate

Run:

```bash
python research_benchmark.py
```

The benchmark currently requires:

```text
F1 >= 0.62
```

## Remaining Limitations

- Validation is currently only on DREAMS, not MASS PSG, because MASS PSG files
  require access approval.
- DREAMS has only 10 excerpts, so results may not generalize.
- The multitaper path is an implementation-inspired candidate extractor, not a
  complete reproduction of MT-KCD.
- The detectors use handcrafted features rather than a deep learning approach.
- Event matching uses an IoU threshold of 0.2; other papers may use different
  matching rules.
- Neural mass simulation and K-complex detection are now connected by
  feature-level and average waveform-level prototypes, but not by robust full
  parameter inversion.
- The thalamocortical model is compact and simplified; it does not yet include
  all detailed intrinsic thalamic currents used in published conductance-based
  models.

## Next Research Steps

1. Improve precision so the window detector beats the DREAMS automatic baseline by a wider margin.
2. Tune event matching criteria and report sensitivity to IoU/onset thresholds.
3. Improve thalamocortical fitting beyond random search and average waveform fitting.
4. Add confidence intervals or bootstrap analysis for per-excerpt variability.
5. Validate on MASS PSG if access becomes available.
6. Reproduce a published MT-KCD or wavelet/TEO method end to end.
