import numpy as np
from neural_mass.models.thalamocortical_model import ThalamocorticalModel, ThalamocorticalParameters, _anti_alias_and_downsample
from neural_mass.detection.event_detection import spindle_detection, mask_segments
from scipy.signal import butter, sosfiltfilt

# Let's try different spindle parameters
# We want to see if we can get non-zero spindle detections on synthetic EEG
# by making spindles transient.
sfreq = 200
dt = 0.001

for spindle_damping in [0.55, 1.2, 2.0]:
    for spindle_drive_offset in [0.45, 0.6, 0.8]:
        for eeg_spindle_weight in [0.18, 0.05, 0.02]:
            p = ThalamocorticalParameters(
                dt=dt,
                neuromodulator_level=0.6,
                spindle_damping=spindle_damping,
                spindle_drive_offset=spindle_drive_offset,
                eeg_spindle_weight=eeg_spindle_weight,
                noise_std=0.015,
            )
            model = ThalamocorticalModel(p, seed=42)
            raw = model.simulate(seconds=60.0)
            eeg = _anti_alias_and_downsample(raw, 1 / dt, sfreq)["eeg"]
            
            # Filter and detect
            mask = spindle_detection(eeg, sampling_frequency=sfreq)
            num_spindles = len(mask_segments(mask))
            
            if num_spindles > 0:
                print(f"FOUND WORKING PARAMS: damping={spindle_damping}, offset={spindle_drive_offset}, weight={eeg_spindle_weight} -> Spindles: {num_spindles}")
                # Print RMS stats
                sos = butter(4, [11, 16], btype="bandpass", output="sos", fs=sfreq)
                filtered = sosfiltfilt(sos, eeg)
                window_size = int(sfreq * 0.2)
                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(filtered, window_shape=window_size)
                windows = np.sqrt(np.mean(windows**2, axis=1))
                print(f"  RMS: Min={np.min(windows):.6f}, Max={np.max(windows):.6f}, Std={np.std(windows):.6f}, Median={np.median(windows):.6f}")
