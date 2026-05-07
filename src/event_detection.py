from numpy.fft import fft, fftfreq
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfilt
import numpy as np

def splindle_detection(signal : NDArray,
                      sampling_frequency : int = 300) -> NDArray:
    """Function to detect splindles

    ARGS:
    -----
        signal : The input signal as a 1D NDarray
        sampling_frequency : The sampling frequency of the signal the default is 300 point per second
    
    RETURNS:
    ------
        A mask with TRUE where the is a Splindle
    """

    #fft_values = fft(signal)
    #frequencies = fftfreq(len(signal))

    # Bandpass the signal
    sos = butter(10, [11,16],
                 btype="bandpass",
                 output="sos",
                 fs=sampling_frequency)
    filtered_signal = sosfilt(sos, signal)

    # Root mean square over sliding windows
    windows = sliding_window_view(filtered_signal,
                                  window_shape=int(sampling_frequency*0.2))
    windows = windows**2
    windows = np.sqrt(np.mean(windows,axis=1))
    
    # Split the signal into windows with high 11 to 16Hz frequency and those with "low"
    threshold = np.quantile(windows,q=0.95)
    mask_out = windows>threshold

    # Loop to filter out sequences that are not long enough
    splindle_sequence_lenght = 0
    true_idxs, potential_idx = [], []
    for idx, is_splindle in enumerate(mask_out):
        if is_splindle:
            splindle_sequence_lenght += 1
            potential_idx.append(idx)
        elif splindle_sequence_lenght > (sampling_frequency * 0.5):
            true_idxs += potential_idx
            potential_idx = []
            splindle_sequence_lenght = 0
        else:
            potential_idx = []
            splindle_sequence_lenght = 0
    
    # We reset the mask now that we know the "true" splindles
    mask_out *= False
    mask_out[true_idxs] = True

    return mask_out

def K_complex_detection(signal : NDArray,
                        sampling_frequency : int = 1000) -> NDArray:
    """
    Function that returns a mask where a K-Complex occurs.
    K-complexes are large, slow waves (>0.5s duration, high amplitude).
    """
    # 1. Low-pass filter below 5Hz to isolate the slow wave
    sos = butter(10, 5,
                 btype="lowpass",
                 output="sos",
                 fs=sampling_frequency)
    filtered_signal = sosfilt(sos, signal)
    
    # 2. Detect peaks that exceed a high amplitude threshold (e.g., 2 standard deviations)
    threshold = 2.0 * np.std(filtered_signal)
    
    mask_out = np.abs(filtered_signal) > threshold
    
    # 3. Filter for duration (must stay above threshold for > 0.5s)
    min_len = int(sampling_frequency * 0.5)
    true_mask = np.zeros_like(mask_out)
    
    count = 0
    for i in range(len(mask_out)):
        if mask_out[i]:
            count += 1
        else:
            if count >= min_len:
                true_mask[i-count:i] = True
            count = 0
            
    return true_mask
