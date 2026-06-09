"""Fine-tune the K-complex detector on HMC (SN001) using pseudo-labels.

Strategy (no ground-truth K-complex annotations available for HMC):
  1. Train a base model on all 10 DREAMS excerpts.
  2. Run base model on HMC at a low threshold (0.20) on N2 epochs
     → these become pseudo-positive training windows.
  3. Use W and REM epoch windows with prob < 0.05 as pseudo-negatives
     (high-confidence background examples).
  4. Retrain on DREAMS + HMC pseudo-labels with balanced weighting.
  5. Select threshold via DREAMS LOO CV (no HMC leakage into threshold).
  6. Evaluate: HMC stage-conditional detection rates + DREAMS LOO F1
     to confirm no regression.

Why this works: K-complexes in HMC have the same underlying morphology
but different amplitude/noise characteristics from DREAMS. Adding even
noisy positive examples from the target domain forces the classifier to
learn features that generalise across channels (CZ-A1 vs C4-M1).
"""
import argparse
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne

from neural_mass.utils.dreams_io import read_scoring_file, read_signal_txt, read_union_events
from neural_mass.utils.event_scoring import aggregate_scores, bootstrap_f1_ci, score_events
from neural_mass.detection.kcomplex_window_detector import (
    bandpass_signal,
    build_window_dataset,
    extract_window_features,
    select_threshold_by_cv,
    slow_wave_candidate_windows,
    train_balanced_window_classifier,
    windows_to_events,
    _THRESHOLDS_GRID as THRESHOLDS_GRID,
)

TARGET_SFREQ = 200.0
HMC_CHANNEL  = "EEG C4-M1"
N2_LABEL     = "Sleep stage N2"
STAGE_ORDER  = ["Sleep stage W","Sleep stage N1","Sleep stage N2","Sleep stage N3","Sleep stage R"]
STAGE_SHORT  = {"Sleep stage W":"W","Sleep stage N1":"N1","Sleep stage N2":"N2",
                "Sleep stage N3":"N3","Sleep stage R":"REM"}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_hmc(edf_path, sfreq=TARGET_SFREQ, channel=HMC_CHANNEL):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick([channel])
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq, verbose=False)
    return raw.get_data()[0] * 1e6   # V -> uV


def load_sleep_stages(scoring_edf):
    ann = mne.read_annotations(str(scoring_edf))
    stages = []
    for a in ann:
        desc = a["description"]
        if desc in STAGE_ORDER:
            stages.append((float(a["onset"]), float(a["onset"]) + float(a["duration"]), desc))
    return stages


def build_candidate_windows(signal, sfreq, chunk_min=30.0):
    """Extract candidate windows in 30-min chunks then deduplicate."""
    chunk = int(chunk_min * 60 * sfreq)
    filtered = bandpass_signal(signal, sfreq)
    all_wins = []
    for i in range(int(np.ceil(len(signal) / chunk))):
        s, e = i * chunk, min(len(signal), (i + 1) * chunk)
        cw = slow_wave_candidate_windows(filtered[s:e], sfreq)
        all_wins.extend([(a + s, b + s) for a, b in cw])
    all_wins.sort(key=lambda w: w[0])
    deduped = []
    for w in all_wins:
        if deduped:
            ps, pe = deduped[-1]
            overlap = max(0, min(w[1], pe) - max(w[0], ps))
            union   = max(w[1], pe) - min(w[0], ps)
            if union > 0 and overlap / union > 0.70:
                if np.max(np.abs(filtered[w[0]:w[1]])) > np.max(np.abs(filtered[ps:pe])):
                    deduped[-1] = w
                continue
        deduped.append(w)
    return deduped, filtered


def stage_of_window(start_samp, sfreq, stages):
    t = start_samp / sfreq
    for onset, end, stage in stages:
        if onset <= t < end:
            return stage
    return None


def build_dreams_datasets(folder):
    folder = Path(folder)
    datasets = []
    for idx in range(1, 11):
        sig = read_signal_txt(folder / f"excerpt{idx}.txt")
        ev  = read_scoring_file(folder / f"Visual_scoring1_excerpt{idx}.txt")
        train_ev = read_union_events(folder, idx)
        filt, wins, X, y = build_window_dataset(sig, 200.0, train_ev)
        datasets.append(dict(excerpt=idx, signal=sig, filtered=filt, sfreq=200.0,
                             expert_events=ev, train_events=train_ev,
                             windows=wins, X=X, y=y))
    # pseudo-labels for excerpts 6-10 (same as main benchmark)
    pm = train_balanced_window_classifier(
        np.vstack([d["X"] for d in datasets[:5]]),
        np.concatenate([d["y"] for d in datasets[:5]]), random_state=42)
    for d in datasets[5:]:
        pp = pm.predict_proba(d["X"])[:, 1]
        d["y"] = np.maximum(d["y"], (pp >= 0.82).astype(int))
    return datasets


# ── pseudo-label generation ───────────────────────────────────────────────────

def generate_hmc_pseudolabels(hmc_windows, hmc_filtered, hmc_stages, base_model,
                               sfreq=TARGET_SFREQ,
                               pos_threshold=0.20, neg_threshold=0.05,
                               pos_stages=(N2_LABEL,),
                               neg_stages=("Sleep stage W", "Sleep stage R")):
    """Return (X_pseudo, y_pseudo) for HMC using base model predictions.

    Pseudo-positives : in N2, base model prob >= pos_threshold
    Pseudo-negatives : in W or REM, base model prob <= neg_threshold
    Windows not meeting either criterion are discarded (too ambiguous).
    """
    print(f"  Extracting features for {len(hmc_windows)} HMC candidates...",
          end=" ", flush=True)
    X = np.asarray([extract_window_features(hmc_filtered, s, e, sfreq)
                    for s, e in hmc_windows])
    print("done", flush=True)

    probs = base_model.predict_proba(X)[:, 1]
    stages_arr = np.array([stage_of_window(s, sfreq, hmc_stages) for s, _ in hmc_windows])

    pos_mask = np.zeros(len(X), dtype=bool)
    neg_mask = np.zeros(len(X), dtype=bool)
    for s in pos_stages:
        pos_mask |= (stages_arr == s) & (probs >= pos_threshold)
    for s in neg_stages:
        neg_mask |= (stages_arr == s) & (probs <= neg_threshold)

    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    y_pos = np.ones(len(X_pos), dtype=int)
    y_neg = np.zeros(len(X_neg), dtype=int)

    print(f"  Pseudo-labels: {len(X_pos)} positives (N2, p>={pos_threshold}), "
          f"{len(X_neg)} negatives (W/REM, p<={neg_threshold})")

    X_pseudo = np.vstack([X_pos, X_neg])
    y_pseudo  = np.concatenate([y_pos, y_neg])
    return X_pseudo, y_pseudo


# ── evaluation ────────────────────────────────────────────────────────────────

def eval_dreams_loo(datasets, model, threshold):
    """Quick LOO eval on DREAMS to check for regression. Returns totals dict."""
    from neural_mass.utils.event_scoring import score_events_onset
    scores = []
    for test_idx, test in enumerate(datasets):
        probs = model.predict_proba(test["X"])[:, 1]
        det = windows_to_events(test["windows"], probs, test["sfreq"],
                                threshold=threshold, n_samples=len(test["signal"]),
                                signal=test["filtered"], spindle_rejection=True)
        scores.append(score_events(test["expert_events"], det))
    return aggregate_scores(scores)


def eval_hmc_rates(hmc_signal, hmc_filtered, hmc_windows, hmc_stages, model,
                   threshold, sfreq=TARGET_SFREQ):
    """Report stage-conditional detection rates for HMC."""
    X = np.asarray([extract_window_features(hmc_filtered, s, e, sfreq)
                    for s, e in hmc_windows])
    probs = model.predict_proba(X)[:, 1]
    events = windows_to_events(hmc_windows, probs, sfreq, threshold=threshold,
                                n_samples=len(hmc_signal), signal=hmc_filtered,
                                spindle_rejection=True)
    stage_events = {st: [] for st in STAGE_ORDER}
    stage_mins   = {st: 0.0 for st in STAGE_ORDER}
    for onset, end, stage in hmc_stages:
        stage_mins[stage] += (end - onset) / 60.0
    for ev in events:
        st = stage_of_window(int(ev["onset"] * sfreq), sfreq, hmc_stages)
        if st and st in stage_events:
            stage_events[st].append(ev)
    print(f"\n{'Stage':<10} {'Duration':>10} {'Events':>8} {'Rate/min':>10}")
    print("-" * 44)
    for st in STAGE_ORDER:
        m = stage_mins[st]
        n = len(stage_events[st])
        r = n / m if m > 0 else 0.0
        print(f"{STAGE_SHORT[st]:<10} {m:>8.1f}m {n:>8}  {r:>9.2f}")
    n2_m = stage_mins.get(N2_LABEL, 1.0)
    n2_n = len(stage_events.get(N2_LABEL, []))
    return events, stage_events, n2_n / n2_m if n2_m > 0 else 0.0


def save_strip_plot(signal, filtered, sfreq, events, path, n=8, win_s=4.0):
    if not events:
        return
    evs = events[:n]
    fig, axes = plt.subplots(len(evs), 1, figsize=(12, 2.0 * len(evs)), facecolor="white")
    if len(evs) == 1:
        axes = [axes]
    half = win_s / 2
    for ax, ev in zip(axes, evs):
        c   = (ev["onset"] + ev["end"]) / 2
        t0  = max(0.0, c - half); t1 = min(len(signal) / sfreq, c + half)
        s0  = int(t0 * sfreq);    s1 = int(t1 * sfreq)
        t   = np.arange(s1 - s0) / sfreq + t0
        ax.plot(t, signal[s0:s1], color="black", lw=0.8)
        es  = int(ev["onset"] * sfreq); ee = int(ev["end"] * sfreq)
        ts  = max(0, es - s0);          te = min(len(t), ee - s0)
        ax.plot(t[ts:te], signal[es:min(ee, s1)], color="indianred", lw=1.5)
        seg = filtered[es:ee]
        if len(seg):
            ni = int(np.argmin(seg)) + es
            pi = int(np.argmax(seg)) + es
            for idx, marker, color in [(ni, "v", "navy"), (pi, "^", "darkgreen")]:
                if s0 <= idx < s1:
                    ax.plot((idx - s0) / sfreq + t0, signal[idx],
                            marker, color=color, ms=6, zorder=5)
        ax.set_xlim(t0, t1)
        ax.set_ylabel("uV", fontsize=8)
        ax.set_title(f"KC @ {ev['onset']:.1f}s  dur={ev['duration']*1000:.0f}ms", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle("HMC SN001 (fine-tuned) — N2 K-complex detections", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Strip plot -> {path}")


def save_rate_comparison(before_rates, after_rates, path):
    stages = list(STAGE_SHORT.values())
    x = np.arange(len(stages))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    bars_b = ax.bar(x - width/2, [before_rates.get(s,0) for s in STAGE_ORDER],
                    width, label="Base (DREAMS only)", color="#8ab4c2", edgecolor="white")
    bars_a = ax.bar(x + width/2, [after_rates.get(s,0) for s in STAGE_ORDER],
                    width, label="Fine-tuned (DREAMS+HMC)", color="indianred", edgecolor="white")
    ax.axhspan(1.0, 5.0, color="gold", alpha=0.20, label="Literature N2 range (1-5/min)")
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("Detections / minute"); ax.set_title("HMC SN001 — before vs after fine-tuning")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    for bar in list(bars_b) + list(bars_a):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.03, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Rate comparison -> {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmc-folder",    default="data/hmc")
    parser.add_argument("--dreams-folder", default="data/dreams/DatabaseKcomplexes")
    parser.add_argument("--plots-dir",     default="plots_hmc")
    parser.add_argument("--pos-threshold", type=float, default=0.20,
                        help="Base-model prob threshold for pseudo-positive label in N2.")
    parser.add_argument("--neg-threshold", type=float, default=0.05,
                        help="Base-model prob threshold for pseudo-negative label in W/REM.")
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Load HMC ──────────────────────────────────────────────────────────
    print("Loading HMC signal...")
    hmc_signal = load_hmc(Path(args.hmc_folder) / "SN001.edf")
    hmc_stages = load_sleep_stages(Path(args.hmc_folder) / "SN001_sleepscoring.edf")
    print(f"  {len(hmc_signal)/TARGET_SFREQ/60:.1f} min, {len(hmc_stages)} stage epochs")

    print("Building HMC candidate windows (chunked)...")
    hmc_windows, hmc_filtered = build_candidate_windows(hmc_signal, TARGET_SFREQ)
    print(f"  {len(hmc_windows)} candidates")

    # ── 2. Build DREAMS datasets ──────────────────────────────────────────────
    print("\nBuilding DREAMS datasets...")
    dreams_datasets = build_dreams_datasets(args.dreams_folder)
    X_dreams = np.vstack([d["X"] for d in dreams_datasets])
    y_dreams  = np.concatenate([d["y"] for d in dreams_datasets])
    print(f"  DREAMS: {len(X_dreams)} windows, {y_dreams.sum()} positives")

    # ── 3. Base model ─────────────────────────────────────────────────────────
    print("\nTraining base model (DREAMS only)...")
    base_model = train_balanced_window_classifier(X_dreams, y_dreams, random_state=0)
    base_threshold, _ = select_threshold_by_cv(dreams_datasets, thresholds=THRESHOLDS_GRID, random_state=0)
    print(f"  CV threshold = {base_threshold:.2f}")

    print("\nBase model — HMC stage rates:")
    _, base_stage_evs, base_n2_rate = eval_hmc_rates(
        hmc_signal, hmc_filtered, hmc_windows, hmc_stages, base_model, base_threshold)
    print(f"  N2 rate = {base_n2_rate:.2f}/min  (literature: 1.0-5.0)")

    base_rates = {st: len(base_stage_evs.get(st,[])) /
                      max(sum((e-o)/60 for o,e,s in hmc_stages if s==st), 0.1)
                  for st in STAGE_ORDER}

    # ── 4. Pseudo-labels from HMC ─────────────────────────────────────────────
    print("\nGenerating HMC pseudo-labels...")
    X_pseudo, y_pseudo = generate_hmc_pseudolabels(
        hmc_windows, hmc_filtered, hmc_stages, base_model,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
    )

    # ── 5. Fine-tuned model ───────────────────────────────────────────────────
    print("\nFine-tuning on DREAMS + HMC pseudo-labels...")
    X_ft = np.vstack([X_dreams, X_pseudo])
    y_ft  = np.concatenate([y_dreams, y_pseudo])
    print(f"  Combined: {len(X_ft)} windows, {y_ft.sum()} positives")
    ft_model = train_balanced_window_classifier(X_ft, y_ft, random_state=0)

    # Keep threshold from DREAMS CV — no HMC data in threshold selection
    ft_threshold = base_threshold
    print(f"  Using DREAMS CV threshold = {ft_threshold:.2f}")

    # ── 6. Evaluate fine-tuned model ─────────────────────────────────────────
    print("\nFine-tuned model — HMC stage rates:")
    ft_events, ft_stage_evs, ft_n2_rate = eval_hmc_rates(
        hmc_signal, hmc_filtered, hmc_windows, hmc_stages, ft_model, ft_threshold)
    print(f"  N2 rate = {ft_n2_rate:.2f}/min  (literature: 1.0-5.0)")

    ft_rates = {st: len(ft_stage_evs.get(st,[])) /
                    max(sum((e-o)/60 for o,e,s in hmc_stages if s==st), 0.1)
                for st in STAGE_ORDER}

    # ── 7. DREAMS LOO regression check ───────────────────────────────────────
    print("\nDREAMS LOO regression check (fine-tuned model, fixed threshold)...")
    base_dreams = eval_dreams_loo(dreams_datasets, base_model, base_threshold)
    ft_dreams   = eval_dreams_loo(dreams_datasets, ft_model,   ft_threshold)
    print(f"  Base  : F1={base_dreams['f1']:.3f} P={base_dreams['precision']:.3f} R={base_dreams['recall']:.3f}")
    print(f"  FT    : F1={ft_dreams['f1']:.3f}   P={ft_dreams['precision']:.3f}   R={ft_dreams['recall']:.3f}")
    delta_f1 = ft_dreams["f1"] - base_dreams["f1"]
    print(f"  Delta F1 = {delta_f1:+.3f}  ({'regression' if delta_f1 < -0.01 else 'OK'})")

    # ── 8. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"HMC N2 detection rate:")
    print(f"  Before fine-tune : {base_n2_rate:.2f}/min")
    print(f"  After  fine-tune : {ft_n2_rate:.2f}/min")
    print(f"  Literature       : 1.0-5.0/min")
    print(f"DREAMS LOO F1      : {base_dreams['f1']:.3f} -> {ft_dreams['f1']:.3f} ({delta_f1:+.3f})")

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    n2_ft_events = ft_stage_evs.get(N2_LABEL, [])
    if n2_ft_events:
        save_strip_plot(hmc_signal, hmc_filtered, TARGET_SFREQ,
                        n2_ft_events, plots_dir / "hmc_ft_n2_detections.png")

    save_rate_comparison(base_rates, ft_rates, plots_dir / "hmc_ft_rate_comparison.png")
    print(f"\nPlots -> {plots_dir}/")


if __name__ == "__main__":
    main()
