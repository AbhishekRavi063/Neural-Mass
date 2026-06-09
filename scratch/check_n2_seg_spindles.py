import numpy as np
from neural_mass import build_neuromodulator_schedule
from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters, _anti_alias_and_downsample
from neural_mass.detection.event_detection import spindle_detection, mask_segments
from scipy.signal import welch, butter, sosfiltfilt

sfreq = 200
dt = 0.001
stage_sequence = [
    ("n1",  60.0),
    ("n2",  180.0),
    ("n3",  120.0),
    ("n2",  90.0),
    ("rem", 60.0),
]
total_seconds = sum(dur for _, dur in stage_sequence)
WARMUP_S = 60.0
warmup_stage = [("n1", WARMUP_S)]
warmup_sched = build_neuromodulator_schedule(warmup_stage, dt=dt, transition_seconds=5.0)
schedule_main = build_neuromodulator_schedule(stage_sequence, dt=dt, transition_seconds=15.0)
schedule = np.concatenate([warmup_sched, schedule_main])
full_seconds = WARMUP_S + total_seconds

params = ThalamocorticalParameters(dt=dt, noise_std=0.015)
full_model = ThalamocorticalModel(params, seed=7)
raw = full_model.simulate(seconds=full_seconds, neuromodulator_schedule=schedule)
cycle_full = _anti_alias_and_downsample(raw, 1 / dt, sfreq)["eeg"]
warmup_samples = int(WARMUP_S * sfreq)
cycle_eeg = cycle_full[warmup_samples:]

n2_seg = cycle_eeg[int(60 * sfreq):int(240 * sfreq)]

# Let's inspect the bandpass filtered n2_seg in the spindle band (11-16 Hz)
sos = butter(4, [11, 16], btype="bandpass", output="sos", fs=sfreq)
filtered = sosfiltfilt(sos, n2_seg)

window_size = int(sfreq * 0.2)
from numpy.lib.stride_tricks import sliding_window_view
windows = sliding_window_view(filtered, window_shape=window_size)
windows = np.sqrt(np.mean(windows**2, axis=1))

threshold = np.median(windows) + 1.5 * np.std(windows)
print("n2_seg length:", len(n2_seg))
print("Filtered Std:", np.std(filtered))
print("Median of windows:", np.median(windows))
print("Std of windows:", np.std(windows))
print("Threshold:", threshold)
print("Max window value:", np.max(windows))
print("Number of windows above threshold:", np.sum(windows > threshold))

mask = spindle_detection(n2_seg, sampling_frequency=sfreq)
print("Detected spindles:", len(mask_segments(mask)))
