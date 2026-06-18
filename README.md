# Sleep EEG — two projects

This repository holds **two independent, side-by-side projects** for sleep EEG
research. They were originally one codebase and have been split so each is a
complete, self-contained project with its own package, dependencies, tests, and
documentation.

| Project | What it does | Package |
|---|---|---|
| [**kcomplex-detector/**](kcomplex-detector) | Machine-learning K-complex detection, validated on DREAMS and stress-tested on HMC | `kcomplex_detector` |
| [**neural-mass-model/**](neural-mass-model) | Neural mass models (Jansen-Rit, thalamocortical) for simulating EEG and fitting parameters with Optuna | `neural_mass` |

Each project installs and runs on its own:

```bash
pip install -e ./kcomplex-detector ".[dev]"     # detector
pip install -e ./neural-mass-model ".[dev]"     # simulator
```

See each project's own `README.md` for full details, datasets, methods, and
results.

## Other folders

- `examples/` — cross-cutting demos that use **both** projects together
  (full pipeline, clinical demo, fitting detected K-complexes to the model).
  These require both packages installed.
- `presentation/` — figures prepared for presentation/LinkedIn.
- `data/` — raw datasets (DREAMS, HMC); not committed, see detector README.

## Headline results

- **DREAMS K-complex detection:** F1 = 0.632 (beats the published 0.620
  baseline, and is more than 2× the 0.301 inter-rater human ceiling).
- **HMC cross-dataset:** no K-complex labels exist, so results are *plausible
  but unverifiable*; pseudo-label fine-tuning brings the N2 detection rate to
  1.08/min, within the literature range of 1–5/min.

Supervised by Jean-Baptiste Chaudron.
