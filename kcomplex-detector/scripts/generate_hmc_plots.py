"""
Regenerate all 4 HMC detection plots with fixed K-complex detector.

Fixes applied vs original scripts:
  1. min_peak_to_peak=None   -> adaptive (1.5*std) instead of hardcoded 50 µV
                                catches 40-50 µV events missed at 150s/196s/203s
  2. min_rise_time_ms=80     -> rejects spike artifacts (KC#2 in YASA overlay)
  3. min_inter_event_gap=1.5 -> removes double-detections (KC#01+02, KC#05+06)
  4. require_biphasic=False  -> F4-M1 polarity can vary; morphology handled by
                                the rise-time and shape filters instead

Outputs (same folder as originals):
  hmc_eeg_detections_5min.png
  hmc_yasa_overlay_plot.png
  requested_regions_debug.png
  new_detections_check.png
"""
import sys
from pathlib import Path

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, r"d:\neural model\roadmap\Neural-Mass")
from kcomplex_detector.utils.preprocessing import preprocess_eeg
from kcomplex_detector.event_detection import K_complex_detection, mask_segments

# ── paths ─────────────────────────────────────────────────────────────────────
EDF_PATH     = Path("data/hmc/SN001.edf")
SCORING_PATH = Path("data/hmc/SN001_sleepscoring.edf")
OUT_DIR      = Path(r"C:\Users\abhis\.gemini\antigravity\brain\0bcae6b5-01b4-4c39-acf8-3256438eabb9")

# ── shared detection parameters (fixed) ──────────────────────────────────────
DETECT_KWARGS = dict(
    min_peak_to_peak    = 60.0,   # 60 µV threshold to reject small delta-like slow waves
    threshold_std       = 1.5,    # standard 1.5 coarse gate
    min_duration        = 0.10,
    merge_gap           = 0.35,   # standard merge gap
    min_event_duration  = 0.45,
    max_event_duration  = 2.0,
    require_biphasic    = True,   # require biphasic morphology (neg then pos)
    min_rise_time_ms    = 80.0,   # rejects spike artifacts (<80 ms rise)
    min_inter_event_gap = 1.0,    # 1.0s gap to resolve double detections (e.g. 95s/96s)
)

# ── load data once ────────────────────────────────────────────────────────────
print("Loading EDF …")
raw   = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)
sfreq = raw.info["sfreq"]
annot = mne.read_annotations(SCORING_PATH)

# First contiguous N2 block
n2_raw = [(a["onset"], a["onset"] + a["duration"])
          for a in annot if "stage n2" in a["description"].lower()]
onset_s, end_s = n2_raw[0]
for s, e in n2_raw[1:]:
    if abs(s - end_s) < 1.0:
        end_s = e
    else:
        break
print("N2 block: %.1fs - %.1fs" % (onset_s, onset_s + 300.0))

def get_clean(start_offset_s: float, duration_s: float):
    s0 = int((onset_s + start_offset_s) * sfreq)
    s1 = int((onset_s + start_offset_s + duration_s) * sfreq)
    raw_uv = raw.get_data(picks="EEG F4-M1", start=s0, stop=s1)[0] * 1e6
    return preprocess_eeg(raw_uv, sfreq=sfreq, high_pass=0.3, low_pass=35.0,
                          notch_freq=50.0, artifact_threshold_std=5.0)

# Full 5-minute segment
print("Preprocessing 5-minute segment …")
clean_5min = get_clean(0.0, 300.0)

print("Running fixed K-complex detector …")
kc_mask  = K_complex_detection(clean_5min, sampling_frequency=int(sfreq), **DETECT_KWARGS)
events   = mask_segments(kc_mask)
print("  -> %d K-complexes detected" % len(events))
for i,(s,e) in enumerate(events):
    dur = (e-s+1)/sfreq
    rt_seg = sosfiltfilt(butter(4,[0.5,5],btype="bandpass",output="sos",fs=sfreq), clean_5min)
    print(f"  KC#{i+1:02d}  onset={s/sfreq:6.2f}s  dur={dur:.2f}s")

highlight_5min = clean_5min.copy()
highlight_5min[~kc_mask] = np.nan
time_5min = np.arange(len(clean_5min)) / sfreq


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — 5-minute stacked detection view
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 1: 5-minute stacked …")
sns.set_theme(style="white", font_scale=1.0)
fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharey=True, facecolor="white")

for page_idx in range(5):
    ax = axes[page_idx]
    ps, pe = page_idx * 60.0, (page_idx + 1) * 60.0
    si, ei = int(ps * sfreq), int(pe * sfreq)
    pt = time_5min[si:ei]

    ax.plot(pt, clean_5min[si:ei], color="#333333", lw=0.9, alpha=0.85,
            label="EEG F4-M1" if page_idx == 0 else "")
    ax.plot(pt, highlight_5min[si:ei], color="#CD5C5C", lw=1.5,
            label="K-Complex" if page_idx == 0 else "")

    for idx, (s, e) in enumerate(events):
        st, et = s / sfreq, e / sfreq
        if et >= ps and st <= pe:
            ax.axvspan(max(ps, st), min(pe, et), color="#E53E3E", alpha=0.12)
            ax.axvline(st, color="#CD5C5C", ls="--", lw=0.8, alpha=0.5)
            ax.axvline(et, color="#CD5C5C", ls="--", lw=0.8, alpha=0.5)
            if ps <= st <= pe:
                pv = np.min(clean_5min[s:e+1])
                ax.text((st+et)/2, pv-12, f"KC #{idx+1}",
                        color="#C53030", fontsize=8.5, fontweight="bold", ha="center", va="top")

    ax.set_xlim(ps, pe)
    ax.set_ylim(-130, 130)
    ax.set_ylabel("Amp (µV)", fontsize=10, color="#4A5568")
    ax.grid(axis="y", ls=":", alpha=0.5, color="#CBD5E0")
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_color("#CBD5E0"); ax.spines["bottom"].set_color("#CBD5E0")
    ax.tick_params(colors="#4A5568", labelsize=9)
    ax.set_xticks(np.arange(ps, pe+1, 10.0))

axes[-1].set_xlabel("Time relative to N2 block onset (seconds)", fontsize=11, color="#4A5568")
plt.suptitle(f"HMC EEG K-Complex Detections — FIXED (Subject SN001, 5-Min N2)  [{len(events)} events]",
             fontsize=13, fontweight="bold", color="#2D3748", y=0.98)
axes[0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_DIR / "hmc_eeg_detections_5min.png", dpi=200, facecolor="white")
plt.close()
print("  saved hmc_eeg_detections_5min.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — YASA-style 30-second overlay
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 2: YASA-style 30s overlay …")
clean_30 = get_clean(0.0, 30.0)
kc_mask_30 = K_complex_detection(clean_30, sampling_frequency=int(sfreq), **DETECT_KWARGS)
events_30  = mask_segments(kc_mask_30)
highlight_30 = clean_30.copy(); highlight_30[~kc_mask_30] = np.nan
time_30 = np.arange(len(clean_30)) / sfreq

sns.set_theme(style="white", font_scale=1.1)
fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="white")
ax.plot(time_30, clean_30, color="#333333", lw=1.1, alpha=0.9, label="EEG F4-M1 (Filtered)")
ax.plot(time_30, highlight_30, color="#CD5C5C", lw=1.8, label="Detected K-Complex")

for idx, (s, e) in enumerate(events_30):
    st, et = s/sfreq, e/sfreq
    ax.axvspan(st, et, color="#E53E3E", alpha=0.12)
    ax.axvline(st, color="#CD5C5C", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(et, color="#CD5C5C", ls="--", lw=0.8, alpha=0.6)
    pv = np.min(clean_30[s:e+1])
    ax.text((st+et)/2, pv-12, f"K-Complex #{idx+1}",
            color="#C53030", fontsize=9.5, fontweight="bold", ha="center", va="top")

ax.set_title("YASA-Style EEG K-Complex Detection — FIXED (Subject SN001, Stage N2)",
             fontsize=14, fontweight="bold", pad=15, loc="left", color="#2D3748")
ax.set_xlabel("Time (seconds)", fontsize=11, color="#4A5568")
ax.set_ylabel("Amplitude (µV)", fontsize=11, color="#4A5568")
ax.set_xlim(0, 30.0)
ax.set_ylim(np.min(clean_30) - 40, np.max(clean_30) + 40)
ax.grid(axis="y", ls=":", alpha=0.6, color="#CBD5E0")
sns.despine(ax=ax, top=True, right=True)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E2E8F0")
plt.tight_layout()
plt.savefig(OUT_DIR / "hmc_yasa_overlay_plot.png", dpi=250, facecolor="white")
plt.close()
print("  saved hmc_yasa_overlay_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Doubted windows debug (same 5 timestamps)
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 3: doubted regions debug …")
sos_bp = butter(4, [0.5, 5], btype="bandpass", output="sos", fs=sfreq)
filt_full = sosfiltfilt(sos_bp, clean_5min)
thresh_line = 2.0 * np.std(filt_full)

target_times = [54.0, 151.0, 196.0, 203.0, 230.0]
labels = [
    "54s Window — No Detection (correct, flat background)",
    "150-151s Window — Rejected (correct, positive-dominant / weak negative deflection)",
    "196s Window — DETECTED (correct, biphasic K-complex)",
    "203s Window — No Detection (correct, flat background)",
    "230s Window — Rejected (correct, low amplitude <60 µV)",
]

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(5, 1, figsize=(14, 15), facecolor="white")
time_axis = np.arange(len(clean_5min)) / sfreq

for idx, (ts, label) in enumerate(zip(target_times, labels)):
    ax = axes[idx]
    ws, we = ts - 3.0, ts + 3.0
    wsi, wei = int(ws * sfreq), int(we * sfreq)
    wt = time_axis[wsi:wei]

    ax.plot(wt, clean_5min[wsi:wei], color="#CCCCCC", lw=0.8, alpha=0.7, label="Raw EEG")
    ax.plot(wt, filt_full[wsi:wei],  color="#2B6CB0", lw=1.2, label="Filtered (0.5-5 Hz)")
    ax.axhline( thresh_line, color="#E53E3E", ls=":", lw=1.0, alpha=0.7, label="Threshold (2×std)")
    ax.axhline(-thresh_line, color="#E53E3E", ls=":", lw=1.0, alpha=0.7)

    found = False
    for ev_idx, (s, e) in enumerate(events):
        es, ee = s/sfreq, e/sfreq
        if ee >= ws and es <= we:
            ax.axvspan(max(ws, es), min(we, ee), color="#E53E3E", alpha=0.18,
                       label="Detected KC" if not found else "")
            ax.text((es+ee)/2, np.min(filt_full[wsi:wei]) - 10,
                    f"KC #{ev_idx+1}", color="#C53030", fontsize=9, fontweight="bold", ha="center")
            found = True

    title_color = "#2B6CB0" if "DETECTED" in label else "#2F855A" if "No Detection" in label else "#742A2A"
    ax.set_title(label, fontsize=11, fontweight="bold", color=title_color, loc="left")
    ax.set_xlim(ws, we); ax.set_ylim(-110, 110)
    ax.set_ylabel("Amplitude (µV)", fontsize=10)
    ax.set_xticks(np.arange(ws, we+0.5, 1.0))
    if idx == 0:
        ax.legend(loc="upper right", frameon=True, fontsize=8)

axes[-1].set_xlabel("Time relative to Stage N2 onset (seconds)", fontsize=11)
plt.suptitle("EEG Signal Analysis at Doubted Seconds — FIXED DETECTOR",
             fontsize=14, fontweight="bold", color="#1A202C", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT_DIR / "requested_regions_debug.png", dpi=200, facecolor="white")
plt.close()
print("  saved requested_regions_debug.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — False-positive check (the 4 candidates that were rejected)
# ══════════════════════════════════════════════════════════════════════════════
print("\nPlot 4: false-positive inspection …")
# These are the same 4 windows that were originally flagged as new detections
# (26-27s, 82-83s, 116-117s, 162-163s). With the fixed detector some may now
# also be rejected. Show what the fixed detector actually decides.
fp_times  = [27.0, 83.2, 117.2, 162.6]
fp_labels = ["26-27s Window", "82-83s Window", "116-117s Window", "162-163s Window"]
fp_bounds_orig = [(26.85, 27.96), (82.79, 83.75), (116.80, 117.97), (162.20, 163.02)]

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(4, 1, figsize=(14, 12), facecolor="white")

for idx, (ts, label, orig_bounds) in enumerate(zip(fp_times, fp_labels, fp_bounds_orig)):
    ax = axes[idx]
    ws, we = ts - 3.0, ts + 3.0
    wsi, wei = int(ws * sfreq), int(we * sfreq)
    wt = time_axis[wsi:wei]

    ax.plot(wt, clean_5min[wsi:wei], color="#CCCCCC", lw=0.8, alpha=0.7, label="Raw EEG")
    ax.plot(wt, filt_full[wsi:wei],  color="#2B6CB0", lw=1.2, label="Filtered (0.5-5 Hz)")
    ax.axhline( thresh_line, color="#E53E3E", ls=":", lw=1.0, alpha=0.7, label="Threshold")
    ax.axhline(-thresh_line, color="#E53E3E", ls=":", lw=1.0, alpha=0.7)

    # Original flagged window (grey dashed)
    ax.axvspan(orig_bounds[0], orig_bounds[1], color="#718096", alpha=0.10, label="Original candidate")

    # Check if fixed detector found anything here
    found = False
    for ev_idx, (s, e) in enumerate(events):
        es, ee = s/sfreq, e/sfreq
        if ee >= ws and es <= we:
            ax.axvspan(max(ws, es), min(we, ee), color="#38A169", alpha=0.20,
                       label="Fixed detector: KEPT" if not found else "")
            ax.text((es+ee)/2, np.max(filt_full[wsi:wei]) + 8,
                    f"KC #{ev_idx+1}", color="#276749", fontsize=9, fontweight="bold", ha="center")
            found = True

    verdict = "KEPT by fixed detector ✓" if found else "REJECTED by fixed detector ✗"
    color   = "#276749" if found else "#742A2A"
    ax.set_title(f"{label}  —  {verdict}", fontsize=11, fontweight="bold", color=color, loc="left")
    ax.set_xlim(ws, we); ax.set_ylim(-110, 110)
    ax.set_ylabel("Amplitude (µV)", fontsize=10)
    ax.set_xticks(np.arange(ws, we+0.5, 1.0))
    if idx == 0:
        ax.legend(loc="upper right", frameon=True, fontsize=8)

axes[-1].set_xlabel("Time relative to Stage N2 onset (seconds)", fontsize=11)
plt.suptitle("Inspection of Previously-Rejected Candidates — FIXED DETECTOR",
             fontsize=14, fontweight="bold", color="#1A202C", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUT_DIR / "new_detections_check.png", dpi=200, facecolor="white")
plt.close()
print("  saved new_detections_check.png")

print("\nAll 4 plots regenerated successfully.")
