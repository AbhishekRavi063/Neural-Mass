# Neural Mass Model

Neural mass models for **simulating sleep EEG** and **fitting model parameters**
to target signals with Optuna.

This is one of two side-by-side projects in this repository. The other,
[`kcomplex-detector`](../kcomplex-detector), is the ML K-complex detector and is
fully independent of this one.

---

## Installation

```bash
cd neural-mass-model
pip install -e ".[dev]"     # dev extra adds pytest, matplotlib
```

## Quick start

```python
from neural_mass import ThalamocorticalSleepModel

model = ThalamocorticalSleepModel(neuromodulator_level=1.0, seed=7)
signals = model.simulate(seconds=30.0)     # NREM-like EEG

from neural_mass import JansenRitModel
jr = JansenRitModel()
jr.fit(target_eeg, n_trials=100)           # Optuna parameter search
print(jr.A_, jr.B_)
```

## Models

- **Jansen-Rit** — classic two-population cortical oscillator (parameters A, B).
- **Thalamocortical sleep model** — compact multi-state model with a
  `neuromodulator_level` knob that moves the output across sleep stages and
  produces grapho-elements (spindles, K-complex-like transients).
- **Spatiotemporal thalamocortical** — spatial extension of the above.

## Parameter fitting and the choice of objective function

The fitting routines deliberately use **different objective functions** matched
to what is being fit:

| Routine | Objective | Why |
|---|---|---|
| `JansenRitModel.fit` | plain RMSE vs target signal | Only 2 params; goal is to reproduce the raw waveform directly. |
| `fit_thalamocortical_features` | normalized-feature MSE | Features (power, counts, ratios) live on very different scales; normalizing stops the largest-magnitude feature from dominating. |
| `fit_thalamocortical_waveform` | standardized-waveform MSE, sign-invariant | The model has no fixed polarity, so we z-score and take `min(direct, flipped)` error to avoid penalizing a correct-but-inverted solution. |
| `fit_thalamocortical_multi_objective` | 3 separate objectives (spectral cosine, statistical, grapho-element) | Keeps competing physiological criteria as a Pareto front instead of merging them with arbitrary weights; the final pick uses L1 compromise programming. |

The common thread: the loss is chosen to match the thing being compared (raw
signal → RMSE; heterogeneous features → normalization; ambiguous polarity →
sign-invariance; competing properties → multi-objective).

## Repository layout

```
neural-mass-model/
├── neural_mass/              the package
│   ├── models/               jansen-rit, thalamocortical, spatiotemporal, graph
│   ├── inference/            inference.py, thalamocortical_fitting.py (Optuna)
│   ├── signal/               vendored event-detection primitives (see note)
│   └── utils/                metrics, preprocessing
├── examples/                 runnable demos
└── tests/                    unit tests
```

**Note on `signal/`** — the fitting code measures spindle/K-complex *rates* in
its own simulated output to match observed grapho-element rates. The low-level
event-detection routines it needs are vendored here as a small self-contained
copy, so this project does not depend on the `kcomplex-detector` project.

## Running tests

```bash
pytest tests/ -q
```

## Notes / TODO

See [NOTES.md](NOTES.md). Possible future directions (suggested by the project
guide): fit literature default parameters, match spindle-production rate to
observed rate, and multichannel fitting (one spectral objective per electrode
sharing thalamic parameters).
