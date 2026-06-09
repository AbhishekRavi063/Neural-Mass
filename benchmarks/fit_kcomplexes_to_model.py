"""Fit the thalamocortical model to detected K-complex windows on DREAMS.

This bridges the two halves of the library:
  1. K-complex detector finds events in real EEG
  2. Thalamocortical model is fitted to those events' spectral features
  3. Parameters are compared to background (non-K-complex) windows

Scientific question: do fitted parameters during K-complexes differ from
background sleep in the direction biophysics predicts?
  Expected: stronger cortex→thalamus drive, higher reticular inhibition,
  lower background drive (DOWN state) during K-complex windows.

Usage:
    python benchmarks/fit_kcomplexes_to_model.py --folder data/dreams/DatabaseKcomplexes
"""
import argparse
from pathlib import Path

import numpy as np

from neural_mass.utils.dreams_io import read_scoring_file, read_signal_txt
from neural_mass.detection.kcomplex_window_detector import (
    bandpass_signal,
    build_window_dataset,
    train_balanced_window_classifier,
    windows_to_events,
)
from neural_mass.inference.thalamocortical_fitting import (
    FEATURE_NAMES,
    extract_window_features,
    fit_thalamocortical_features,
    parameter_difference,
)

SFREQ = 200.0
MIN_WINDOW_SAMPLES = int(0.5 * SFREQ)   # at least 0.5 s for a meaningful fit


def extract_event_windows(signal, events, sfreq, context_pad=0.10):
    """Return raw EEG segments for a list of detected events."""
    segments = []
    pad = int(round(context_pad * sfreq))
    for ev in events:
        start = max(0, int(round(ev["onset"] * sfreq)) - pad)
        end = min(len(signal), int(round(ev["end"] * sfreq)) + pad)
        if end - start >= MIN_WINDOW_SAMPLES:
            segments.append(signal[start:end])
    return segments


def background_windows(signal, detected_events, sfreq, n_windows=20, window_seconds=1.5):
    """Sample windows from quiet background (not overlapping any detected event)."""
    n_samples = len(signal)
    win_len = int(round(window_seconds * sfreq))
    detected_mask = np.zeros(n_samples, dtype=bool)
    for ev in detected_events:
        s = max(0, int(round(ev["onset"] * sfreq)) - win_len)
        e = min(n_samples, int(round(ev["end"] * sfreq)) + win_len)
        detected_mask[s:e] = True

    rng = np.random.default_rng(42)
    candidates = np.arange(0, n_samples - win_len, win_len // 2)
    quiet = [c for c in candidates if not detected_mask[c:c + win_len].any()]
    if len(quiet) > n_windows:
        idx = rng.choice(len(quiet), n_windows, replace=False)
        quiet = [quiet[i] for i in sorted(idx)]
    return [signal[c:c + win_len] for c in quiet]


def aggregate_features(windows, sfreq):
    """Median feature vector across a list of EEG windows."""
    if not windows:
        return None
    feature_list = [extract_window_features(w, sfreq) for w in windows]
    return {name: float(np.median([f[name] for f in feature_list])) for name in FEATURE_NAMES}


def run_fitting(
    folder="data/dreams/DatabaseKcomplexes",
    n_fit_trials=40,
    max_excerpts=5,
):
    """Detect K-complexes on DREAMS, fit thalamocortical model to events vs background."""
    folder = Path(folder)
    excerpts = list(range(1, min(11, max_excerpts + 1)))

    all_signals = []
    all_expert_events = []
    datasets = []

    print(f"Loading {len(excerpts)} excerpts...")
    for idx in excerpts:
        signal = read_signal_txt(folder / f"excerpt{idx}.txt")
        expert_events = read_scoring_file(folder / f"Visual_scoring1_excerpt{idx}.txt")
        filtered, windows, X, y = build_window_dataset(signal, SFREQ, expert_events)
        all_signals.append(signal)
        all_expert_events.append(expert_events)
        datasets.append({
            "idx": idx, "signal": signal, "filtered": filtered,
            "sfreq": SFREQ, "expert_events": expert_events,
            "windows": windows, "X": X, "y": y,
        })

    # Train on all excerpts together (we're not evaluating detection here,
    # we just need good detections to extract K-complex windows from)
    X_all = np.vstack([d["X"] for d in datasets])
    y_all = np.concatenate([d["y"] for d in datasets])
    model = train_balanced_window_classifier(X_all, y_all, random_state=42)

    all_kc_windows = []
    all_bg_windows = []

    for d in datasets:
        probs = model.predict_proba(d["X"])[:, 1]
        detected = windows_to_events(
            d["windows"], probs, SFREQ,
            threshold=0.50,
            n_samples=len(d["signal"]),
            signal=d["filtered"],
            spindle_rejection=True,
        )
        kc_segs = extract_event_windows(d["signal"], detected, SFREQ)
        bg_segs = background_windows(d["signal"], detected, SFREQ)
        all_kc_windows.extend(kc_segs)
        all_bg_windows.extend(bg_segs)
        print(
            f"  excerpt={d['idx']}  detected={len(detected)}  "
            f"kc_windows={len(kc_segs)}  bg_windows={len(bg_segs)}"
        )

    print(f"\nTotal K-complex windows: {len(all_kc_windows)}")
    print(f"Total background windows: {len(all_bg_windows)}")

    if not all_kc_windows:
        print("No K-complex windows extracted — cannot fit.")
        return

    kc_features = aggregate_features(all_kc_windows, SFREQ)
    bg_features = aggregate_features(all_bg_windows, SFREQ)

    print("\n--- Feature comparison ---")
    print(f"{'Feature':<25} {'K-complex':>12} {'Background':>12} {'Ratio':>8}")
    print("-" * 60)
    for name in FEATURE_NAMES:
        kc_val = kc_features[name]
        bg_val = bg_features[name]
        ratio = kc_val / (bg_val + 1e-10)
        marker = " *" if abs(ratio - 1.0) > 0.5 else ""
        print(f"  {name:<23} {kc_val:12.4f} {bg_val:12.4f} {ratio:8.2f}x{marker}")

    print(f"\nFitting thalamocortical model to K-complex features ({n_fit_trials} trials)...")
    kc_params, kc_fitted_features, kc_error = fit_thalamocortical_features(
        kc_features, seconds=10.0, sfreq=int(SFREQ), n_trials=n_fit_trials, seed=7
    )

    print(f"Fitting thalamocortical model to background features ({n_fit_trials} trials)...")
    bg_params, bg_fitted_features, bg_error = fit_thalamocortical_features(
        bg_features, seconds=10.0, sfreq=int(SFREQ), n_trials=n_fit_trials, seed=7
    )

    diff = parameter_difference(kc_params, bg_params)

    print("\n--- Fitted thalamocortical parameters ---")
    print(f"{'Parameter':<30} {'K-complex':>12} {'Background':>12} {'Delta (KC-BG)':>12}")
    print("-" * 68)
    for name, delta in diff.items():
        kc_val = getattr(kc_params, name)
        bg_val = getattr(bg_params, name)
        direction = "+ [UP]" if delta > 0 else "- [DOWN]"
        print(f"  {name:<28} {kc_val:12.4f} {bg_val:12.4f} {delta:+12.4f} {direction}")

    print(f"\nFit error - K-complex: {kc_error:.4f}   Background: {bg_error:.4f}")

    print("\n--- Biological interpretation ---")
    _interpret(diff)

    return kc_params, bg_params, diff


def _interpret(diff):
    """Print plain-English interpretation of parameter differences."""
    notes = []
    if diff.get("cortex_to_thalamus", 0) > 0.05:
        notes.append("  + Stronger cortex->thalamus drive during K-complexes "
                     "(cortex actively initiating thalamic suppression)")
    if diff.get("reticular_inhibition", 0) > 0.05:
        notes.append("  + Higher reticular inhibition during K-complexes "
                     "(thalamic gating is active - consistent with sleep protection)")
    if diff.get("background_drive", 0) < -0.05:
        notes.append("  - Lower background drive during K-complexes "
                     "(DOWN state - reduced tonic excitation)")
    if diff.get("adaptation_strength", 0) > 0.05:
        notes.append("  + Stronger adaptation during K-complexes "
                     "(neural fatigue following the large discharge)")
    if not notes:
        notes.append("  Parameter differences small - "
                     "may need more excerpts or longer fitting.")
    for note in notes:
        print(note)


def main():
    parser = argparse.ArgumentParser(
        description="Fit thalamocortical model to detected K-complex vs background windows."
    )
    parser.add_argument("--folder", default="data/dreams/DatabaseKcomplexes")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Optuna trials per fit (default 40).")
    parser.add_argument("--max-excerpts", type=int, default=5,
                        help="How many DREAMS excerpts to use (default 5, max 10).")
    args = parser.parse_args()
    run_fitting(args.folder, n_fit_trials=args.n_trials, max_excerpts=args.max_excerpts)


if __name__ == "__main__":
    main()
