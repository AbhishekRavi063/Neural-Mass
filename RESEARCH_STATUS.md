# Research Status

This project is still a research prototype, but it now has a reproducible
validation path for K-complex detection and basic tests for the neural mass
simulation layer.

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

## Regression Gate

Run:

```bash
python research_benchmark.py
```

The benchmark currently requires:

```text
F1 >= 0.58
```

## Remaining Limitations

- Validation is currently only on DREAMS, not MASS PSG, because MASS PSG files
  require access approval.
- DREAMS has only 10 excerpts, so results may not generalize.
- The multitaper path is an implementation-inspired candidate extractor, not a
  complete reproduction of MT-KCD.
- The classifier uses handcrafted features rather than a deep learning approach.
- Event matching uses an IoU threshold of 0.2; other papers may use different
  matching rules.
- Neural mass simulation and K-complex detection are still separate workflows.

## Next Research Steps

1. Compare against the DREAMS automatic detection files.
2. Add second-expert comparison for the first 5 DREAMS excerpts.
3. Tune event matching criteria and report sensitivity to IoU threshold.
4. Validate on MASS PSG if access becomes available.
5. Reproduce a published MT-KCD or wavelet/TEO method end to end.
