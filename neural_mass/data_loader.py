import mne
import numpy as np
from mne.datasets.sleep_physionet.age import fetch_data

def load_sample_sleep_data(subject=0, recording=[1]):
    """
    Downloads and loads real sleep EEG data from PhysioNet.
    Subject 0, Recording 1 is a healthy volunteer.
    """
    print(f"Fetching PhysioNet data for subject {subject}...")
    # This downloads the files (approx 5MB)
    data_files = fetch_data(subjects=[subject], recording=recording)
    
    # Path to the PSG (The waves) and Hypnogram (The sleep stage labels)
    psg_path = data_files[0][0]
    hypno_path = data_files[0][1]
    
    # Load the raw EEG
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=True)
    
    # Load the annotations (labels)
    annotations = mne.read_annotations(hypno_path)
    raw.set_annotations(annotations, emit_warning=False)
    
    # We typically want the 'EEG Fpz-Cz' channel (The forehead sensor)
    raw.pick_channels(['EEG Fpz-Cz'])
    
    return raw

def get_sleep_stage_data(raw, stage="Sleep stage 2"):
    """
    Extracts segments of a specific sleep stage.
    Sleep Stage 2 is where most Spindles and K-complexes happen.
    """
    # Find where the stage starts and ends
    events, event_id = mne.events_from_annotations(raw, verbose=True)
    
    # Create epochs (30-second windows)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=30, 
                        baseline=None, preload=True, verbose=True)
    
    if stage in event_id:
        return epochs[stage].get_data(copy=True)[:, 0, :] # Returns [Windows, TimePoints]
    else:
        print(f"Stage {stage} not found in this recording.")
        return None

def generate_realistic_eeg(duration=30, sfreq=100, include_k_complexes=True, seed=None, return_events=False):
    """
    Generates a realistic mock EEG signal with 1/f noise, 
    low-frequency drift, simulated sleep spindles, and optional K-complexes.
    """
    rng = np.random.default_rng(seed)
    n_pts = int(duration * sfreq)
    times = np.arange(n_pts) / sfreq
    
    # 1. 1/f noise (Brownian noise)
    noise = np.cumsum(rng.normal(0, 1, n_pts))
    noise = (noise - np.mean(noise)) / np.std(noise) * 5 # Scale to ~5uV
    
    # 2. Low-frequency drift (0.1Hz)
    drift = 10 * np.sin(2 * np.pi * 0.1 * times)
    
    # 3. Simulate a few Spindles (12Hz)
    spindle = np.zeros(n_pts)
    spindle_mask = np.zeros(n_pts, dtype=bool)
    # Put a spindle at 5s and 15s
    for start_t in [5, 15]:
        idx = int(start_t * sfreq)
        dur = int(1.5 * sfreq) # 1.5 second spindle
        envelope = np.hanning(dur)
        wave = np.sin(2 * np.pi * 12 * times[idx:idx+dur])
        spindle[idx:idx+dur] = envelope * wave * 15 # 15uV amplitude
        spindle_mask[idx:idx+dur] = True

    # 4. Simulate K-complexes: a sharp negative deflection followed by a slower positive rebound.
    k_complex = np.zeros(n_pts)
    k_complex_mask = np.zeros(n_pts, dtype=bool)
    if include_k_complexes:
        for start_t in [9, 22]:
            idx = int(start_t * sfreq)
            dur = int(1.1 * sfreq)
            if idx + dur > n_pts:
                continue
            local_t = np.linspace(0, 1, dur)
            negative = -50 * np.exp(-((local_t - 0.28) ** 2) / (2 * 0.08 ** 2))
            positive = 28 * np.exp(-((local_t - 0.62) ** 2) / (2 * 0.16 ** 2))
            k_complex[idx:idx+dur] += negative + positive
            k_complex_mask[idx:idx+dur] = True

    signal = noise + drift + spindle + k_complex
    if return_events:
        return signal, {
            "spindle": spindle_mask,
            "k_complex": k_complex_mask,
        }
    return signal
