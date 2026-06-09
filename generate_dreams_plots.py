"""
YASA-style K-complex plots + FN diagnostic for the DREAMS dataset.

Plot style mirrors YASA plot_detection:
  - Raw EEG in black, lw=1
  - Detected segment redrawn over it in colour (not a fill, the signal itself)
  - Landmark triangles at NegPeak (down) and PosPeak (up)
  - No bandpass overlay
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
from scipy.signal import butter, sosfiltfilt, find_peaks, welch
from pathlib import Path
sys.path.insert(0, '.')

from neural_mass.utils.dreams_io import read_scoring_file, read_signal_txt, read_union_events
from neural_mass.detection.kcomplex_window_detector import (
    build_window_dataset, train_balanced_window_classifier,
    select_threshold_by_cv, windows_to_events, _THRESHOLDS_GRID,
)
from neural_mass.utils.event_scoring import score_events

FOLDER = Path('data/dreams/DatabaseKcomplexes')
SFREQ  = 200.0
OUT    = Path('plots_dreams')
OUT.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def bpfilt(sig, lo, hi, fs=SFREQ, order=4):
    sos = butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, sig)

def event_iou(a, b):
    a0 = a['onset'];  a1 = a.get('end', a.get('end_time', a0+1.0))
    b0 = b['onset'];  b1 = b.get('end', b.get('end_time', b0+1.0))
    ov = max(0.0, min(a1,b1)-max(a0,b0))
    un = max(a1,b1)-min(a0,b0)
    return ov/un if un>0 else 0.0

def classify(detected, expert):
    tps, fps, fns = [], [], []
    matched = set()
    for d in detected:
        best, bj = 0.0, -1
        for j,e in enumerate(expert):
            v = event_iou(d, e)
            if v > best: best, bj = v, j
        if best >= 0.20: tps.append(d); matched.add(bj)
        else:            fps.append(d)
    for j,e in enumerate(expert):
        if j not in matched: fns.append(e)
    return tps, fps, fns

def landmark(ev, filt_sig):
    """Return (neg_peak_time, pos_peak_time) in seconds."""
    s = int(ev['onset']*SFREQ)
    e = int(ev.get('end', ev.get('end_time', ev['onset']+1.2))*SFREQ)
    s, e = max(0,s), min(len(filt_sig),e)
    seg = filt_sig[s:e]
    if len(seg) < 4: return ev['onset'], ev.get('end', ev['onset']+0.5)
    ni = s + int(np.argmin(seg))
    pi = s + int(np.argmax(seg[int(np.argmin(seg)):]))+int(np.argmin(seg))
    return ni/SFREQ, pi/SFREQ

# ── load & LOO ───────────────────────────────────────────────────────────────
datasets = []
for idx in range(1, 11):
    sig    = read_signal_txt(FOLDER / f'excerpt{idx}.txt')
    expert = read_scoring_file(FOLDER / f'Visual_scoring1_excerpt{idx}.txt')
    train  = read_union_events(FOLDER, idx)
    filt, wins, X, y = build_window_dataset(sig, SFREQ, train)
    datasets.append(dict(excerpt=idx, signal=sig, filtered=filt,
                         sfreq=SFREQ, expert_events=expert,
                         train_events=train, windows=wins, X=X, y=y))
print('Data loaded.')

fold_data = []
for ti, ts in enumerate(datasets):
    tr  = [d for i,d in enumerate(datasets) if i!=ti]
    Xt  = np.vstack([d['X'] for d in tr])
    yt  = np.concatenate([d['y'] for d in tr])
    mdl = train_balanced_window_classifier(Xt, yt, random_state=100+ti)
    thr, _ = select_threshold_by_cv(tr, thresholds=_THRESHOLDS_GRID,
                                    random_state=100+ti)
    probs = mdl.predict_proba(ts['X'])[:,1]
    det   = windows_to_events(ts['windows'], probs, SFREQ,
                               threshold=thr, n_samples=len(ts['signal']),
                               signal=ts['filtered'], spindle_rejection=True)
    sc    = score_events(ts['expert_events'], det)
    fold_data.append({**ts, 'detected':det, 'probs':probs,
                      'threshold':thr, 'score':sc})
print('LOO done.')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 – YASA-style EEG strips
#   raw = black lw=1  |  TP segment = indianred lw=2  |  FP = steelblue lw=2
#   FN expert = dashed orange line on top  |  NegPeak ▼  PosPeak ▲
# ═══════════════════════════════════════════════════════════════════════════
STRIP   = 30        # seconds per strip
STRIPS  = 4         # strips per excerpt
EXCEPTS = [2, 4, 1, 7]   # which excerpts to show

n_rows = len(EXCEPTS) * STRIPS
fig, axes = plt.subplots(n_rows, 1, figsize=(16, n_rows*1.55),
                         gridspec_kw={'hspace':0.05})

legend_items = [
    mlines.Line2D([],[],color='k',       lw=1,   label='Raw EEG'),
    mlines.Line2D([],[],color='indianred',lw=2,  label='Detected  TP'),
    mlines.Line2D([],[],color='steelblue',lw=2,  label='Detected  FP'),
    mlines.Line2D([],[],color='darkorange',lw=2, ls='--', label='Missed  FN (expert)'),
    mlines.Line2D([],[],color='indianred', marker='v', ls='none',
                  ms=5, label='NegPeak'),
    mlines.Line2D([],[],color='indianred', marker='^', ls='none',
                  ms=5, label='PosPeak'),
]

filt_slow_cache = {}
row = 0
for exc in EXCEPTS:
    fd   = fold_data[exc-1]
    raw  = fd['signal']
    if exc not in filt_slow_cache:
        filt_slow_cache[exc] = bpfilt(raw, 0.3, 1.5)
    fsig = filt_slow_cache[exc]
    tps, fps, fns = classify(fd['detected'], fd['expert_events'])
    sc   = fd['score']
    times_full = np.arange(len(raw)) / SFREQ

    for strip_i in range(STRIPS):
        t0 = 60 + strip_i * STRIP      # start at 60s to skip quiet opening
        t1 = t0 + STRIP
        s0, s1 = int(t0*SFREQ), min(len(raw), int(t1*SFREQ))
        t_ax  = times_full[s0:s1]
        ax    = axes[row]

        # raw black
        ax.plot(t_ax, raw[s0:s1], color='k', lw=0.7, zorder=2)

        # YASA style: replot the signal segment in colour where event exists
        def draw_event_seg(ev, color, lw=1.8, zorder=5):
            on  = ev['onset']
            end = ev.get('end', ev.get('end_time', on+1.0))
            if end < t0 or on > t1: return
            es = max(s0, int(on*SFREQ))
            ee = min(s1, int(end*SFREQ))
            if ee <= es: return
            t_seg = times_full[es:ee]
            ax.plot(t_seg, raw[es:ee], color=color, lw=lw, zorder=zorder)

        def draw_landmarks(ev, color):
            on  = ev['onset']
            end = ev.get('end', ev.get('end_time', on+1.0))
            if end < t0 or on > t1: return
            nt, pt = landmark(ev, fsig)
            # NegPeak down-triangle
            if t0 <= nt <= t1:
                ni = int(nt*SFREQ)
                ax.plot(nt, raw[ni], marker='v', color=color,
                        ms=5, zorder=8, ls='none')
            # PosPeak up-triangle
            if t0 <= pt <= t1:
                pi = int(pt*SFREQ)
                ax.plot(pt, raw[pi], marker='^', color=color,
                        ms=5, zorder=8, ls='none')

        for ev in tps:
            draw_event_seg(ev, 'indianred')
            draw_landmarks(ev, 'indianred')
        for ev in fps:
            draw_event_seg(ev, 'steelblue')
            draw_landmarks(ev, 'steelblue')
        # FN: dashed orange line (expert annotation the detector missed)
        for ev in fns:
            on  = ev['onset']
            end = ev.get('end', ev.get('end_time', on+1.0))
            if end < t0 or on > t1: continue
            es = max(s0, int(on*SFREQ))
            ee = min(s1, int(end*SFREQ))
            if ee > es:
                ax.plot(times_full[es:ee], raw[es:ee],
                        color='darkorange', lw=2.0, ls='--', zorder=6)
                # also add onset tick at top
                ymax = ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 100
                ax.annotate('', xy=(on if on>=t0 else t0, raw[es]),
                            xytext=(on if on>=t0 else t0,
                                    raw[es]+abs(raw[es])*0.3+20),
                            arrowprops=dict(arrowstyle='->', color='darkorange',
                                            lw=1.2))

        ax.set_xlim(t0, t1)
        ylim = 1.25 * max(150, np.percentile(np.abs(raw[s0:s1]), 99))
        ax.set_ylim(-ylim, ylim)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, labelsize=7)

        if strip_i == 0:
            ax.set_title(
                f'Excerpt {exc}  |  F1={sc["f1"]:.2f}  '
                f'Prec={sc["precision"]:.2f}  Rec={sc["recall"]:.2f}  |  '
                f'TP={sc["tp"]}  FP={sc["fp"]}  FN={sc["fn"]}',
                fontsize=8.5, fontweight='bold', loc='left', pad=2)
        if row == n_rows-1:
            ax.set_xlabel('Time (s)', fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        row += 1

axes[0].legend(handles=legend_items, loc='upper right', fontsize=7.5,
               framealpha=0.92, ncol=3)
fig.suptitle(
    'DREAMS K-complex Detection  —  YASA-style Signal Marking\n'
    'Black = raw EEG  |  Red = TP  |  Blue = FP  |  Orange dashed = FN (expert, missed)',
    fontsize=11, fontweight='bold', y=1.002)
fig.tight_layout()
fig.savefig(OUT / 'plot1_yasa_marking.png', dpi=150, bbox_inches='tight')
plt.close()
print('Plot 1 saved.')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 – Event-locked average  (YASA plot_average style, clean)
# ═══════════════════════════════════════════════════════════════════════════
WIN_B = int(0.8 * SFREQ)
WIN_A = int(1.2 * SFREQ)
N     = WIN_B + WIN_A

tp_tr, fn_tr, fp_tr = [], [], []

for fd in fold_data:
    raw  = fd['signal']
    fsig = bpfilt(raw, 0.3, 1.5)
    tps, fps, fns = classify(fd['detected'], fd['expert_events'])
    for ev_list, store in [(tps, tp_tr), (fps, fp_tr), (fns, fn_tr)]:
        for ev in ev_list:
            s = int(ev['onset']*SFREQ)
            e = int(ev.get('end', ev['onset']+1.2)*SFREQ)
            seg = fsig[s:e] if e<=len(fsig) else fsig[s:]
            if len(seg)<4: continue
            ni = int(np.argmin(seg))
            center = s + ni
            a, b = center-WIN_B, center+WIN_A
            if a>=0 and b<=len(fsig):
                tr = fsig[a:b].copy()
                tr -= np.mean(tr[:WIN_B//4])
                store.append(tr)

def trim(traces):
    arr = [t for t in traces if len(t)>=N]
    if not arr: return np.empty((0,N))
    return np.array([t[:N] for t in arr])

tp_arr = trim(tp_tr); fn_arr = trim(fn_tr); fp_arr = trim(fp_tr)
t_ax   = np.linspace(-WIN_B/SFREQ, WIN_A/SFREQ, N)

fig, ax = plt.subplots(figsize=(9, 4.5))
for arr, color, label in [
    (tp_arr, 'indianred',  f'TP  (detected)   n={len(tp_arr)}'),
    (fn_arr, 'darkorange', f'FN  (missed)     n={len(fn_arr)}'),
    (fp_arr, 'steelblue',  f'FP  (hallucin.)  n={len(fp_arr)}'),
]:
    if len(arr)==0: continue
    m   = arr.mean(axis=0)
    sem = arr.std(axis=0) / np.sqrt(len(arr))
    ax.fill_between(t_ax, m-sem, m+sem, alpha=0.18, color=color)
    ax.plot(t_ax, m, color=color, lw=2.2, label=label)

ax.axvline(0, color='k', lw=0.8, ls='--')
ax.axhline(0, color='#BDBDBD', lw=0.5)
ax.set_xlabel('Time relative to negative peak  (s)', fontsize=10)
ax.set_ylabel('Amplitude  (0.3-1.5 Hz filtered, uV)', fontsize=10)
ax.set_title('Event-locked Average K-complex  (mean ± SEM, all excerpts)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(-WIN_B/SFREQ, WIN_A/SFREQ)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT / 'plot2_event_locked_avg.png', dpi=150)
plt.close()
print('Plot 2 saved.')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 – FN rejection-reason audit
#   For each expert FN, test each gate manually and record why it was missed
# ═══════════════════════════════════════════════════════════════════════════
# We test each FN against 5 conditions:
#  A. Amplitude  : ptp < 1.5 * local_rolling_std  (adaptive threshold)
#  B. ZCR        : zero-crossing rate on bandpass > 3.0/s
#  C. Template   : corrcoef with ideal KC < 0.15
#  D. Alpha ctx  : alpha/delta > 2.5 in ±20s context
#  E. Not found  : candidate window never generated (prob < threshold)

from scipy.ndimage import uniform_filter1d
from neural_mass.detection.event_detection import _IDEAL_KC, robust_std

def compute_rolling_std(filtered_sig, fs, window_min=5):
    wsamp = int(fs * window_min * 60)
    rs = uniform_filter1d(np.abs(filtered_sig), size=wsamp) * np.sqrt(np.pi/2.0)
    rs = np.maximum(rs, robust_std(filtered_sig) * 0.5)
    return rs

def check_fn_reasons(fd):
    raw  = fd['signal']
    fsig = bpfilt(raw, 0.3, 1.5)
    rs   = compute_rolling_std(fsig, SFREQ)
    tps, fps, fns = classify(fd['detected'], fd['expert_events'])

    reasons = {'amplitude':0, 'zcr':0, 'template':0,
               'alpha_ctx':0, 'not_found':0, 'passed_all':0}
    fn_ptps, tp_ptps = [], []

    # TP amplitudes
    for ev in tps:
        s = int(ev['onset']*SFREQ); e = int(ev.get('end',ev['onset']+1.2)*SFREQ)
        seg = fsig[max(0,s):min(len(fsig),e)]
        if len(seg)>=4: tp_ptps.append(float(np.max(seg)-np.min(seg)))

    for ev in fns:
        s  = int(ev['onset']*SFREQ)
        e  = int(ev.get('end', ev.get('end_time', ev['onset']+1.2))*SFREQ)
        s, e = max(0,s), min(len(fsig),e)
        seg  = fsig[s:e]
        if len(seg)<4: continue
        ptp  = float(np.max(seg)-np.min(seg))
        fn_ptps.append(ptp)
        neg_idx = s + int(np.argmin(seg))
        local_thr = 1.5 * float(rs[neg_idx])

        # A: amplitude
        if ptp < local_thr:
            reasons['amplitude'] += 1; continue

        # B: ZCR on filtered
        mean_seg = np.mean(seg)
        zc  = np.sum(np.diff(np.sign(seg - mean_seg)) != 0)
        dur = len(seg)/SFREQ
        if dur > 0 and zc/dur > 3.0:
            reasons['zcr'] += 1; continue

        # C: template correlation
        if len(seg) >= 8:
            interp = np.interp(np.linspace(0,len(seg)-1,100), np.arange(len(seg)), seg)
            interp = (interp-np.mean(interp))/(np.std(interp)+1e-10)
            tc = float(np.corrcoef(interp, _IDEAL_KC)[0,1])
            if tc < 0.15:
                reasons['template'] += 1; continue

        # D: alpha context
        ctx_r = int(SFREQ*20)
        ctx   = raw[max(0,neg_idx-ctx_r):min(len(raw),neg_idx+ctx_r)]
        if len(ctx) >= int(SFREQ*4):
            fr, ps = welch(ctx, fs=SFREQ, nperseg=min(512,len(ctx)))
            d_pwr  = float(np.sum(ps[(fr>=0.5)&(fr<=4.0)])) + 1e-10
            a_pwr  = float(np.sum(ps[(fr>=8.0)&(fr<=13.0)]))
            if a_pwr/d_pwr > 2.5:
                reasons['alpha_ctx'] += 1; continue

        # E: not found (classifier never fired)
        reasons['not_found'] += 1

    return reasons, fn_ptps, tp_ptps

all_reasons = {'amplitude':0,'zcr':0,'template':0,
               'alpha_ctx':0,'not_found':0,'passed_all':0}
all_fn_ptps, all_tp_ptps = [], []
for fd in fold_data:
    r, fp2, tp2 = check_fn_reasons(fd)
    for k in all_reasons: all_reasons[k] += r[k]
    all_fn_ptps.extend(fp2); all_tp_ptps.extend(tp2)

print('FN reasons:', all_reasons)
print(f'FN median PTP: {np.median(all_fn_ptps):.1f} uV')
print(f'TP median PTP: {np.median(all_tp_ptps):.1f} uV')

# Figure: two side-by-side panels
fig, (ax_pie, ax_amp) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: stacked horizontal bar (cleaner than pie)
labels_r = ['Amplitude\ntoo low', 'ZCR\ntoo high', 'Template\ncorr < 0.15',
            'Alpha context\nhigh (Wake)', 'Classifier\nnot fired']
keys_r   = ['amplitude','zcr','template','alpha_ctx','not_found']
colors_r = ['#E53935','#FB8C00','#8E24AA','#039BE5','#43A047']
vals_r   = [all_reasons[k] for k in keys_r]
total_fn = sum(vals_r)
pcts     = [v/total_fn*100 for v in vals_r]

ypos = np.arange(len(labels_r))
bars = ax_pie.barh(ypos, pcts, color=colors_r, height=0.55, zorder=3)
ax_pie.set_yticks(ypos)
ax_pie.set_yticklabels(labels_r, fontsize=9)
ax_pie.set_xlabel('% of all FN events', fontsize=9)
ax_pie.set_title(f'FN Rejection Reason Audit\n(total FN = {total_fn})',
                 fontweight='bold', fontsize=10)
for bar, pct, v in zip(bars, pcts, vals_r):
    ax_pie.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f'{v}  ({pct:.0f}%)', va='center', fontsize=8.5)
ax_pie.set_xlim(0, max(pcts)*1.35)
ax_pie.grid(axis='x', alpha=0.3, zorder=0)
ax_pie.spines['top'].set_visible(False)
ax_pie.spines['right'].set_visible(False)

# Right: amplitude distributions TP vs FN
bins = np.linspace(0, 350, 35)
ax_amp.hist(all_tp_ptps, bins=bins, alpha=0.55, color='indianred',
            density=True, label=f'TP  (n={len(all_tp_ptps)})  median={np.median(all_tp_ptps):.0f} uV')
ax_amp.hist(all_fn_ptps, bins=bins, alpha=0.55, color='darkorange',
            density=True, label=f'FN  (n={len(all_fn_ptps)})  median={np.median(all_fn_ptps):.0f} uV')
ax_amp.axvline(np.median(all_tp_ptps), color='indianred', lw=1.5, ls='--')
ax_amp.axvline(np.median(all_fn_ptps), color='darkorange', lw=1.5, ls='--')
ax_amp.set_xlabel('Peak-to-peak amplitude  (0.3-1.5 Hz, uV)', fontsize=9)
ax_amp.set_ylabel('Density', fontsize=9)
ax_amp.set_title('Amplitude Distribution: TP vs FN\n(tells us how much overlap exists)',
                 fontweight='bold', fontsize=10)
ax_amp.legend(fontsize=8.5)
ax_amp.spines['top'].set_visible(False)
ax_amp.spines['right'].set_visible(False)
ax_amp.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(OUT / 'plot3_fn_diagnostic.png', dpi=150)
plt.close()
print('Plot 3 saved.')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 – Per-excerpt FN cause stacked bar
# ═══════════════════════════════════════════════════════════════════════════
exc_reasons = []
for fd in fold_data:
    r, _, _ = check_fn_reasons(fd)
    exc_reasons.append((fd['excerpt'], r))

fig, ax = plt.subplots(figsize=(13, 4.5))
x = np.arange(len(exc_reasons))
bottoms = np.zeros(len(exc_reasons))
for k, c, lbl in zip(keys_r, colors_r, labels_r):
    vals = [r[k] for _, r in exc_reasons]
    ax.bar(x, vals, bottom=bottoms, color=c, label=lbl.replace('\n',' '), zorder=3)
    bottoms += np.array(vals, dtype=float)
ax.set_xticks(x)
ax.set_xticklabels([f'Ex {e}' for e,_ in exc_reasons])
ax.set_ylabel('FN event count', fontsize=9)
ax.set_title('FN Rejection Reason per Excerpt  —  what is killing missed detections?',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'plot4_fn_per_excerpt.png', dpi=150)
plt.close()
print('Plot 4 saved.')

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5 – Amplitude threshold vs actual FN amplitudes
#   Shows the rolling threshold curve overlaid on FN dot positions
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False,
                         gridspec_kw={'hspace':0.45})

for ax, exc_num in zip(axes, [2, 4]):
    fd   = fold_data[exc_num-1]
    raw  = fd['signal']
    fsig = bpfilt(raw, 0.3, 1.5)
    rs   = compute_rolling_std(fsig, SFREQ)
    thr_curve = 1.5 * rs
    times_s   = np.arange(len(raw)) / SFREQ

    tps, fps, fns = classify(fd['detected'], fd['expert_events'])

    # Envelope of bandpassed signal
    from scipy.signal import hilbert
    env = np.abs(hilbert(fsig))

    ax.plot(times_s, env,       color='#B0BEC5', lw=0.6, zorder=1, label='Signal envelope')
    ax.plot(times_s, thr_curve, color='k',       lw=1.1, zorder=3, ls='-',
            label='Adaptive threshold (1.5 × rolling std)')

    # TP dots at NegPeak
    for ev in tps:
        s = int(ev['onset']*SFREQ); e = int(ev.get('end',ev['onset']+1.2)*SFREQ)
        seg = fsig[max(0,s):min(len(fsig),e)]
        if len(seg)<2: continue
        ni = s + int(np.argmin(seg))
        ptp = float(np.max(seg)-np.min(seg))
        ax.plot(ni/SFREQ, ptp, marker='o', color='indianred',
                ms=4, ls='none', zorder=5)

    # FN dots
    for ev in fns:
        s = int(ev['onset']*SFREQ); e = int(ev.get('end',ev['onset']+1.2)*SFREQ)
        seg = fsig[max(0,s):min(len(fsig),e)]
        if len(seg)<2: continue
        ni = s + int(np.argmin(seg))
        ptp = float(np.max(seg)-np.min(seg))
        ax.plot(ni/SFREQ, ptp, marker='x', color='darkorange',
                ms=6, mew=1.5, ls='none', zorder=6)

    sc = fd['score']
    ax.set_title(
        f'Excerpt {exc_num}  |  F1={sc["f1"]:.2f}  TP={sc["tp"]}  FP={sc["fp"]}  FN={sc["fn"]}\n'
        'Dots: red circle = TP PTP,  orange x = FN PTP  vs adaptive threshold curve',
        fontsize=9, fontweight='bold', loc='left')
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Peak-to-peak amplitude (uV)', fontsize=9)
    ax.set_ylim(0, 300)
    legend_items2 = [
        mlines.Line2D([],[],color='k',      lw=1.1, label='Adaptive threshold'),
        mlines.Line2D([],[],color='#B0BEC5',lw=1,   label='Signal envelope'),
        mlines.Line2D([],[],color='indianred', marker='o', ls='none', ms=5, label='TP'),
        mlines.Line2D([],[],color='darkorange',marker='x', ls='none', ms=6,
                      mew=1.5, label='FN (missed)'),
    ]
    ax.legend(handles=legend_items2, fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)

fig.suptitle('Adaptive Threshold vs Detected / Missed K-complex Amplitudes',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'plot5_threshold_vs_events.png', dpi=150)
plt.close()
print('Plot 5 saved.')

print(f'\nAll 5 plots -> {OUT.resolve()}')
