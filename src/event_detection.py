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
                        sampling_frequency : int = 30) -> NDArray:
    """Function that return the indexes where a K-Complex occur

    ARGS :
    -----
        signal : The signal as 1D array 
        sampling_frequency : The sampling frequency

    RETURNS:
    ------
        A mask of the places where the K-complex occurs
    """

    sos = butter(10, 5,
                 btype="lowpass",
                 output="sos",
                 fs=sampling_frequency)
    filtered_signal = sosfilt(sos, signal)
