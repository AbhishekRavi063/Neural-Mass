"""Joint multi-stage calibration of the thalamocortical model.

Fits ONE shared parameter set so that sweeping the neuromodulator knob reproduces
the canonical *achievable* sleep signatures of this 2-oscillator model:

    Wake -> low delta, few spindles      (desynchronized)
    N2   -> delta up, spindles PEAK       (sigma bump)
    N3   -> delta maximal, few spindles   (slow-wave sleep)

The model has no alpha generator, so we calibrate the two bands it can actually
control -- delta (slow cortical oscillator) and sigma (spindle oscillator) -- in
the log domain, per stage. Theta/alpha/beta are left to the 1/f background.

Run:  python scripts/calibrate_stages.py --n-trials 200
"""
import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from neural_mass.models.thalamocortical_model import (
    ThalamocorticalModel, ThalamocorticalParameters,
    _anti_alias_and_downsample, _generate_pink_noise,
    _apply_observation_lowpass, _apply_measurement_floor,
)
from neural_mass.inference.thalamocortical_fitting import band_power

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Neuromodulator level per stage (0 = wake, 1 = deep NREM).
STAGES = {"Wake": 0.0, "N2": 0.6, "N3": 1.0}

# Per-stage targets for the two bands this model can control, as fractions of
# 0.5-30 Hz power. Captures: delta rises Wake->N3; sigma (spindles) peaks at N2.
STAGE_TARGETS = {
    "Wake": {"delta": 0.30, "sigma": 0.05},
    "N2":   {"delta": 0.55, "sigma": 0.17},
    "N3":   {"delta": 0.78, "sigma": 0.05},
}

# Shared structural parameters fitted once across all stages.
SHARED_RANGES = {
    "adaptation_strength":       (0.25, 0.85),
    "cortex_to_thalamus":        (0.10, 0.75),
    "thalamus_to_cortex":        (0.08, 0.65),
    "reticular_inhibition":      (0.25, 0.95),
    "background_drive":          (-0.05, 0.45),
    "cortical_damping":          (0.08, 0.35),
    "spindle_damping":           (0.30, 0.90),
    "relay_to_reticular":        (0.30, 1.20),
    "spindle_feedback_gain":     (3.0, 15.0),
    "cortical_excitation_scale": (8.0, 28.0),
    "eeg_spindle_weight":        (0.02, 0.20),
    "spindle_drive_offset":      (0.20, 0.70),
    "cortical_frequency":        (0.40, 1.30),
    "spindle_frequency":         (11.0, 15.5),
    "adaptation_tau":            (1.0, 3.5),
    "eeg_relay_weight":          (0.01, 0.20),
    "neuromodulation_strength":  (0.20, 0.60),
    "pink_noise_std":            (0.003, 0.025),
    "eeg_lowpass_hz":            (16.0, 35.0),
    "measurement_noise_std":     (0.0, 0.05),
}

SFREQ = 200
DT = 0.001


def simulate_stage(values, nm, seconds, burn, seed):
    p = ThalamocorticalParameters(dt=DT, noise_std=0.0, neuromodulator_level=nm, **values)
    model = ThalamocorticalModel(p, seed=seed)
    raw = model.simulate(seconds=seconds)
    eeg = _anti_alias_and_downsample(raw, 1 / DT, SFREQ)["eeg"][int(burn * SFREQ):]
    rng = np.random.default_rng(seed)
    if p.pink_noise_std > 0:
        eeg = eeg + _generate_pink_noise(len(eeg), p.pink_noise_std, rng)
    eeg = _apply_observation_lowpass(eeg, SFREQ, p.eeg_lowpass_hz)
    eeg = _apply_measurement_floor(eeg, p.measurement_noise_std, rng)
    return eeg


def band_fracs(eeg):
    tot = band_power(eeg, SFREQ, 0.5, 30.0) or 1.0
    return {
        "delta": band_power(eeg, SFREQ, 0.5, 4.0) / tot,
        "theta": band_power(eeg, SFREQ, 4.0, 8.0) / tot,
        "alpha": band_power(eeg, SFREQ, 8.0, 12.0) / tot,
        "sigma": band_power(eeg, SFREQ, 12.0, 16.0) / tot,
        "beta":  band_power(eeg, SFREQ, 16.0, 30.0) / tot,
    }


def stage_error(fracs, target):
    # log-domain error on the controllable bands (delta, sigma)
    err = 0.0
    for b, tgt in target.items():
        err += (np.log10(fracs[b] + 1e-6) - np.log10(tgt + 1e-6)) ** 2
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--burn", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="figures/stage_sweep.png")
    ap.add_argument("--params-out", default="figures/calibrated_stage_params.json")
    args = ap.parse_args()

    def objective(trial):
        values = {n: trial.suggest_float(n, lo, hi) for n, (lo, hi) in SHARED_RANGES.items()}
        total = 0.0
        for name, nm in STAGES.items():
            try:
                eeg = simulate_stage(values, nm, args.seconds, args.burn, args.seed)
            except Exception:
                return 1e6
            total += stage_error(band_fracs(eeg), STAGE_TARGETS[name])
        return total

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    print(f"Calibrating shared parameters across {list(STAGES)} ({args.n_trials} trials)...")
    study.optimize(objective, n_trials=args.n_trials)
    best = study.best_params
    print(f"Best joint error = {study.best_value:.4f}\n")

    print(f"{'stage':6}  {'delta':>16}  {'sigma':>16}")
    print(f"{'':6}  {'target / model':>16}  {'target / model':>16}")
    results = {}
    for name, nm in STAGES.items():
        eeg = simulate_stage(best, nm, args.seconds, args.burn, args.seed)
        fr = band_fracs(eeg)
        results[name] = (eeg, fr)
        t = STAGE_TARGETS[name]
        print(f"{name:6}  {t['delta']:.2f} / {fr['delta']:.2f}        "
              f"{t['sigma']:.2f} / {fr['sigma']:.2f}")

    Path(args.params_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.params_out).write_text(json.dumps(best, indent=2))
    print(f"\nSaved calibrated params -> {args.params_out}")

    _plot(results, args.out)


def _plot(results, out):
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    colors = {"Wake": "#7f8c8d", "N2": "#2e86c1", "N3": "#1a5276"}
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.55, wspace=0.25)

    # left: 6 s waveform per stage
    for i, (name, (eeg, fr)) in enumerate(results.items()):
        ax = fig.add_subplot(gs[i, 0])
        win = int(6 * SFREQ)
        seg = eeg[:win]
        seg = (seg - seg.mean()) / (seg.std() or 1.0)
        ax.plot(np.arange(win) / SFREQ, seg, color=colors[name], lw=0.7)
        ax.set_title(name, fontsize=10, loc="left")
        ax.set_ylim(-4, 4); ax.set_ylabel("z")
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")

    # right top: overlaid spectra -- the spindle (sigma) bump is the clear effect
    ax2 = fig.add_subplot(gs[:2, 1])
    for name, (eeg, fr) in results.items():
        f, P = welch(eeg - eeg.mean(), fs=SFREQ, nperseg=min(1024, len(eeg)))
        band = f <= 30
        P = P / P[band].sum()
        lw = 2.2 if name == "N2" else 1.4
        ax2.semilogy(f[band], P[band], color=colors[name], lw=lw, label=name)
    ax2.axvspan(12, 16, color="#e67e22", alpha=0.10)
    ax2.annotate("spindle bump\n(N2 only)", xy=(14, 6e-3), xytext=(19, 3e-2),
                 fontsize=8.5, color="#b9540c",
                 arrowprops=dict(arrowstyle="->", color="#b9540c", lw=1.0))
    ax2.set_xlim(0, 30); ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Normalized power")
    ax2.set_title("Spectrum per stage — spindle (12–16 Hz) appears only at N2",
                  fontsize=9.5, loc="left")
    ax2.legend(fontsize=9, frameon=False); ax2.grid(alpha=0.25, which="both")

    # right bottom: spindle (sigma) band fraction -- the hero result
    ax3 = fig.add_subplot(gs[2, 1])
    names = list(results)
    x = np.arange(len(names))
    sigma = [results[n][1]["sigma"] for n in names]
    bars = ax3.bar(x, sigma, 0.55, color=["#7f8c8d", "#e67e22", "#1a5276"])
    for xi, s in zip(x, sigma):
        ax3.text(xi, s + 0.004, f"{s:.2f}", ha="center", fontsize=8.5)
    ax3.set_xticks(x); ax3.set_xticklabels(names)
    ax3.set_ylabel("spindle (sigma)\npower fraction")
    ax3.set_ylim(0, max(sigma) * 1.35)
    ax3.set_title("Spindle power peaks at N2", fontsize=10, loc="left")

    fig.suptitle("The neuromodulator knob gates sleep spindles (N2 peak)",
                 fontsize=12, fontweight="bold")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved figure -> {out}")


if __name__ == "__main__":
    main()
