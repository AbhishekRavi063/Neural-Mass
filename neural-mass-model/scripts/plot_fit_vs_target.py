"""Simulator figure: thalamocortical model fitted to real sleep EEG.

Shows a real sleep-EEG segment next to the model's best fit, plus a power-spectrum
overlay -- the spectrum is what the fit optimizes, so it is the honest evidence the
model reproduces the signal. Uses `fit_thalamocortical_spectral`, which matches the
delta-dominant broadband spectral shape (the default multi-objective fit collapses
onto the spindle band).

Usage:
    python scripts/plot_fit_vs_target.py \
        --target "../data/dreams/DatabaseKcomplexes/excerpt1.txt"
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

from neural_mass.inference.thalamocortical_fitting import (
    fit_thalamocortical_spectral, simulate_eeg,
)


def load_dreams_txt(path):
    vals = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return np.asarray(vals)


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() or 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="../data/dreams/DatabaseKcomplexes/excerpt1.txt")
    ap.add_argument("--sfreq", type=int, default=200)
    ap.add_argument("--start-sec", type=float, default=480.0,
                    help="offset of a representative N2 window (strongly delta-dominant, modest spindle)")
    ap.add_argument("--fit-seconds", type=float, default=30.0)
    ap.add_argument("--n-trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--burn-sec", type=float, default=10.0)
    ap.add_argument("--sim-sec", type=float, default=60.0)
    ap.add_argument("--out", default="figures/simulator_fit_vs_target.png")
    args = ap.parse_args()

    sfreq = args.sfreq
    full = load_dreams_txt(args.target)
    s0 = int(args.start_sec * sfreq)
    target = full[s0:s0 + int(args.fit_seconds * sfreq)]
    print(f"Target: {len(target)/sfreq:.0f}s of real CZ-A1 sleep EEG")

    print(f"Fitting thalamocortical model to spectral shape ({args.n_trials} trials)...")
    params, profile, err = fit_thalamocortical_spectral(
        target, sfreq=sfreq, n_trials=args.n_trials, seed=args.seed,
        burn_seconds=args.burn_sec,
    )
    print("Fitted band profile:", {k: round(v, 3) for k, v in profile.items()})
    print(f"Spectral-shape error = {err:.4f}")

    burn = int(args.burn_sec * sfreq)
    fitted = simulate_eeg(params, args.burn_sec + args.sim_sec, sfreq, args.seed)[burn:]

    nper = min(1024, len(target), len(fitted))
    f_t, P_t = welch(target - target.mean(), fs=sfreq, nperseg=nper)
    f_f, P_f = welch(fitted - fitted.mean(), fs=sfreq, nperseg=nper)
    band = f_t <= 30
    P_t = P_t / P_t[band].sum()
    P_f = P_f / P_f[band].sum()

    win = int(8 * sfreq)
    t = np.arange(win) / sfreq
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.45, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, zscore(target[:win]), color="#1f5fb4", lw=0.8)
    ax0.set_title("Real sleep EEG  (DREAMS, CZ-A1)", fontsize=10, loc="left")
    ax0.set_ylabel("z-score"); ax0.set_xticklabels([]); ax0.set_ylim(-4, 4)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, zscore(fitted[:win]), color="#c0392b", lw=0.8)
    ax1.set_title("Thalamocortical model  —  best fit", fontsize=10, loc="left")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("z-score"); ax1.set_ylim(-4, 4)

    ax2 = fig.add_subplot(gs[:, 1])
    ax2.semilogy(f_t[band], P_t[band], color="#1f5fb4", lw=1.6, label="Real EEG")
    ax2.semilogy(f_f[band], P_f[band], color="#c0392b", lw=1.6, label="Fitted model")
    ax2.set_title("Power spectrum  (the fit target)", fontsize=10, loc="left")
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Normalized power")
    ax2.set_xlim(0, 30); ax2.legend(fontsize=9, frameon=False)
    ax2.grid(alpha=0.25, which="both")

    fig.suptitle("Thalamocortical fit: model reproduces the spectral profile of real sleep EEG",
                 fontsize=11, fontweight="bold")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
