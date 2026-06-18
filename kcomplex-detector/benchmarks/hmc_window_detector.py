"""Cross-dataset evaluation of the window K-complex detector on HMC (SN001).

The HMC dataset has no K-complex annotations, so this is a sanity/generalization check:
  - Train on all 10 DREAMS excerpts (Expert 1+2 union labels)
  - Apply to HMC SN001 C4-M1 channel (resampled to 200 Hz)
  - Report detection rate per sleep stage (should be high in N2, low elsewhere)
  - Save YASA-style plots of example N2 detections

Expected literature range: 1–5 K-complexes/minute in N2 sleep.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne

from kcomplex_detector.utils.dreams_io import read_scoring_file, read_signal_txt, read_union_events
from kcomplex_detector.kcomplex_window_detector import (
    bandpass_signal,
    build_window_dataset,
    train_balanced_window_classifier,
    windows_to_events,
    select_threshold_by_cv,
)
from kcomplex_detector.kcomplex_window_detector import _THRESHOLDS_GRID as THRESHOLDS_GRID


TARGET_SFREQ = 200.0
HMC_CHANNEL = "EEG C4-M1"
N2_LABEL = "Sleep stage N2"

STAGE_ORDER = ["Sleep stage W", "Sleep stage N1", "Sleep stage N2",
               "Sleep stage N3", "Sleep stage R"]
STAGE_SHORT = {"Sleep stage W": "W", "Sleep stage N1": "N1",
               "Sleep stage N2": "N2", "Sleep stage N3": "N3",
               "Sleep stage R": "REM"}


def load_hmc_signal(edf_path, channel=HMC_CHANNEL, target_sfreq=TARGET_SFREQ):
    """Load a single EEG channel from HMC EDF, resample to target_sfreq."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick([channel])
    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    signal = raw.get_data()[0] * 1e6  # V → µV
    return signal, float(target_sfreq)


def load_sleep_stages(scoring_edf_path):
    """Return list of (onset_s, duration_s, stage_str) from HMC scoring EDF."""
    ann = mne.read_annotations(str(scoring_edf_path))
    stages = []
    for a in ann:
        desc = a["description"]
        if desc in STAGE_ORDER:
            stages.append((float(a["onset"]), float(a["duration"]), desc))
    return stages


def epochs_for_stage(stages, target_stage):
    """Return list of (onset_s, end_s) for all epochs of a given stage."""
    return [(on, on + dur) for on, dur, st in stages if st == target_stage]


def train_dreams_model(dreams_folder):
    """Train on all 10 DREAMS excerpts. Returns (model, threshold, datasets)."""
    folder = Path(dreams_folder)
    datasets = []
    for idx in range(1, 11):
        signal = read_signal_txt(folder / f"excerpt{idx}.txt")
        expert_events = read_scoring_file(folder / f"Visual_scoring1_excerpt{idx}.txt")
        train_events = read_union_events(folder, idx)
        filtered, windows, X, y = build_window_dataset(signal, 200.0, train_events)
        datasets.append({
            "excerpt": idx, "signal": signal, "filtered": filtered,
            "sfreq": 200.0, "expert_events": expert_events,
            "train_events": train_events, "windows": windows, "X": X, "y": y,
        })

    # Self-training pseudo-labels for excerpts 6-10 (same as benchmark)
    pseudo_train = [d for d in datasets if d["excerpt"] <= 5]
    X_pseudo = np.vstack([d["X"] for d in pseudo_train])
    y_pseudo = np.concatenate([d["y"] for d in pseudo_train])
    pseudo_model = train_balanced_window_classifier(X_pseudo, y_pseudo, random_state=42)
    for d in datasets:
        if d["excerpt"] >= 6:
            pseudo_probs = pseudo_model.predict_proba(d["X"])[:, 1]
            d["y"] = np.maximum(d["y"], (pseudo_probs >= 0.82).astype(int))

    # Train final model on all 10 excerpts
    X_all = np.vstack([d["X"] for d in datasets])
    y_all = np.concatenate([d["y"] for d in datasets])
    model = train_balanced_window_classifier(X_all, y_all, random_state=0)

    # CV threshold selection across all excerpts
    best_threshold, _ = select_threshold_by_cv(datasets, thresholds=THRESHOLDS_GRID, random_state=0)
    return model, best_threshold, datasets


def detect_full_recording(signal, sfreq, model, threshold, chunk_minutes=30.0):
    """Detect K-complexes on the full signal using overlapping 30-min chunks.

    Why chunks: slow_wave_candidate_windows has max_candidates=1000 (tuned for
    30-min DREAMS excerpts). Running on a 427-min recording in one pass caps at
    1000 candidates total and triggers the adaptive noise escalation, both of
    which cause severe under-detection. Processing in 30-min chunks (matching
    the DREAMS training distribution) keeps the local noise estimate and
    candidate cap per-chunk, giving ~1000 candidates per chunk.

    Features are extracted with absolute indices into the full filtered signal,
    so long-context features (±20-30s) see the correct surrounding data even
    at chunk boundaries.
    """
    from kcomplex_detector.kcomplex_window_detector import slow_wave_candidate_windows, extract_window_features

    chunk_samples = int(chunk_minutes * 60 * sfreq)
    print(f"  Bandpass filtering full recording...", end=" ", flush=True)
    filtered = bandpass_signal(signal, sfreq)
    print("done", flush=True)

    all_windows = []
    total_chunks = int(np.ceil(len(signal) / chunk_samples))
    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_samples
        end = min(len(signal), start + chunk_samples)
        chunk_filt = filtered[start:end]
        chunk_wins = slow_wave_candidate_windows(chunk_filt, sfreq)
        # Shift window indices to absolute positions in the full recording
        all_windows.extend([(s + start, e + start) for s, e in chunk_wins])

    # Global deduplication: drop windows with >70% overlap (cross-chunk duplicates)
    all_windows.sort(key=lambda w: w[0])
    deduped = []
    for w in all_windows:
        if deduped:
            ps, pe = deduped[-1]
            overlap = max(0, min(w[1], pe) - max(w[0], ps))
            union = max(w[1], pe) - min(w[0], ps)
            if union > 0 and overlap / union > 0.70:
                if np.max(np.abs(filtered[w[0]:w[1]])) > np.max(np.abs(filtered[ps:pe])):
                    deduped[-1] = w
                continue
        deduped.append(w)

    print(f"  {total_chunks} chunks -> {len(deduped)} candidate windows", flush=True)
    if not deduped:
        return [], filtered

    print(f"  Extracting features (full-recording context)...", end=" ", flush=True)
    X = np.asarray([extract_window_features(filtered, s, e, sfreq) for s, e in deduped])
    print("done", flush=True)

    probs = model.predict_proba(X)[:, 1]
    events = windows_to_events(
        deduped, probs, sfreq,
        threshold=threshold,
        n_samples=len(signal),
        signal=filtered,
        spindle_rejection=True,
    )
    return events, filtered


def filter_events_by_stage(events, stage_intervals):
    """Keep only events whose onset falls within one of the stage intervals."""
    result = []
    for ev in events:
        for onset_s, end_s in stage_intervals:
            if onset_s <= ev["onset"] < end_s:
                result.append(ev)
                break
    return result


def save_strip_plot(signal, filtered_signal, sfreq, events, out_path, n_examples=6, window_s=4.0):
    """Save a YASA-style strip of example K-complex detections."""
    if not events:
        print("No events to plot.")
        return
    sample_events = events[:n_examples]
    n = len(sample_events)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.0 * n), facecolor="white")
    if n == 1:
        axes = [axes]
    half = window_s / 2
    filtered = filtered_signal
    for ax, ev in zip(axes, sample_events):
        center_s = (ev["onset"] + ev["end"]) / 2
        t0 = max(0.0, center_s - half)
        t1 = min(len(signal) / sfreq, center_s + half)
        s0 = int(t0 * sfreq); s1 = int(t1 * sfreq)
        t = np.arange(s1 - s0) / sfreq + t0
        ax.plot(t, signal[s0:s1], color="black", lw=0.8, label="raw")
        # Highlight detected event
        ev_s = int(ev["onset"] * sfreq); ev_e = int(ev["end"] * sfreq)
        ax.plot(t[max(0, ev_s - s0):min(len(t), ev_e - s0)],
                signal[ev_s:min(ev_e, s1)], color="indianred", lw=1.5, label="KC")
        # Negative peak marker
        seg_filt = filtered[ev_s:ev_e]
        if len(seg_filt):
            neg_idx = int(np.argmin(seg_filt)) + ev_s
            pos_idx = int(np.argmax(seg_filt)) + ev_s
            if s0 <= neg_idx < s1:
                ax.plot((neg_idx - s0) / sfreq + t0, signal[neg_idx], "v",
                        color="navy", ms=6, zorder=5)
            if s0 <= pos_idx < s1:
                ax.plot((pos_idx - s0) / sfreq + t0, signal[pos_idx], "^",
                        color="darkgreen", ms=6, zorder=5)
        ax.set_xlim(t0, t1)
        ax.set_ylabel("µV", fontsize=8)
        ax.set_title(f"KC @ {ev['onset']:.1f}s  dur={ev['duration']*1000:.0f}ms", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle("HMC SN001 — Detected K-complexes (N2 epochs)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Strip plot saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmc-folder", default="data/hmc")
    parser.add_argument("--dreams-folder", default="data/dreams/DatabaseKcomplexes")
    parser.add_argument("--plots-dir", default="plots_hmc")
    args = parser.parse_args()

    hmc_folder = Path(args.hmc_folder)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(exist_ok=True)

    print("Loading HMC signal...")
    signal, sfreq = load_hmc_signal(hmc_folder / "SN001.edf")
    print(f"  {len(signal)/sfreq/60:.1f} min at {sfreq} Hz  ({len(signal)} samples)")

    print("Loading sleep stages...")
    stages = load_sleep_stages(hmc_folder / "SN001_sleepscoring.edf")
    for stage in STAGE_ORDER:
        eps = epochs_for_stage(stages, stage)
        total_min = sum(e - o for o, e in eps) / 60
        print(f"  {STAGE_SHORT[stage]:4s}  {len(eps):4d} epochs  {total_min:.1f} min")

    print("\nTraining on DREAMS (all 10 excerpts)...")
    model, threshold, _ = train_dreams_model(args.dreams_folder)
    print(f"  CV threshold = {threshold:.2f}")

    print("\nRunning detector on full recording (required for correct long-context features)...")
    all_events, filtered_signal = detect_full_recording(signal, sfreq, model, threshold)
    print(f"  Total detections across recording: {len(all_events)}")

    print("\nAssigning events to sleep stages...")
    stage_results = {}
    for stage in STAGE_ORDER:
        eps = epochs_for_stage(stages, stage)
        total_min = sum(e - o for o, e in eps) / 60
        evs = filter_events_by_stage(all_events, eps)
        rate = len(evs) / total_min if total_min > 0 else 0.0
        stage_results[stage] = {"events": evs, "total_min": total_min, "rate": rate}
        print(f"  {STAGE_SHORT[stage]:4s}: {len(evs):4d} events in {total_min:.1f} min -> {rate:.2f}/min")

    print("\n--- Stage-conditional detection rates ---")
    print(f"{'Stage':<10} {'Duration':>10} {'Events':>8} {'Rate/min':>10}")
    print("-" * 44)
    for stage in STAGE_ORDER:
        r = stage_results.get(stage, {})
        mins = r.get("total_min", 0)
        evs = len(r.get("events", []))
        rate = r.get("rate", 0.0)
        print(f"{STAGE_SHORT[stage]:<10} {mins:>8.1f}m {evs:>8}  {rate:>9.2f}")

    n2_events = stage_results.get(N2_LABEL, {}).get("events", [])
    print(f"\nN2 K-complex rate: {len(n2_events)/stage_results[N2_LABEL]['total_min']:.2f} /min")
    print(f"Literature range: 1.0–5.0 /min")

    # Example strip plot from N2 detections
    if n2_events:
        save_strip_plot(signal, filtered_signal, sfreq, n2_events, plots_dir / "hmc_n2_detections.png")

    # Bar chart: rate per stage
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    short_labels = [STAGE_SHORT[s] for s in STAGE_ORDER if s in stage_results]
    rates = [stage_results[s].get("rate", 0.0) for s in STAGE_ORDER if s in stage_results]
    colors = ["#e07b54" if s == N2_LABEL else "#8ab4c2" for s in STAGE_ORDER if s in stage_results]
    bars = ax.bar(short_labels, rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhspan(1.0, 5.0, color="gold", alpha=0.25, label="Literature N2 range (1–5/min)")
    ax.set_ylabel("Detections / minute", fontsize=10)
    ax.set_title("HMC SN001 — K-complex detection rate by sleep stage", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rate:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "hmc_stage_rates.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nRate chart -> {plots_dir / 'hmc_stage_rates.png'}")


if __name__ == "__main__":
    main()
