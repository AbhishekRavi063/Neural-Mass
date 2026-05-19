import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from src.thalamocortical_model import simulate_thalamocortical_sleep


def band_power(signal, sfreq, low, high):
    frequencies, power = welch(signal - np.mean(signal), fs=sfreq, nperseg=min(1024, len(signal)))
    band = (frequencies >= low) & (frequencies <= high)
    return float(np.sum(power[band]))


def dominant_frequency(signal, sfreq, low, high):
    frequencies, power = welch(signal - np.mean(signal), fs=sfreq, nperseg=min(1024, len(signal)))
    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band):
        return 0.0
    band_frequencies = frequencies[band]
    band_power_values = power[band]
    return float(band_frequencies[np.argmax(band_power_values)])


def main():
    sfreq = 200
    signals = simulate_thalamocortical_sleep(seconds=30, sampling_frequency=sfreq, seed=21)
    eeg = signals["eeg"]
    cortical = signals["cortical_pyramidal"]
    relay = signals["thalamic_relay"]
    reticular = signals["thalamic_reticular"]
    spindle = signals["spindle"]
    times = np.arange(len(eeg)) / sfreq

    slow_power = band_power(eeg, sfreq, 0.3, 1.5)
    spindle_power = band_power(eeg, sfreq, 11.0, 16.0)
    slow_peak = dominant_frequency(eeg, sfreq, 0.3, 1.5)
    spindle_peak = dominant_frequency(eeg, sfreq, 11.0, 16.0)

    print("==========================================")
    print(" THALAMOCORTICAL NREM-LIKE MODEL DEMO")
    print("==========================================")
    print(f"Samples: {len(eeg)}")
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Slow-band peak: {slow_peak:.2f} Hz")
    print(f"Spindle-band peak: {spindle_peak:.2f} Hz")
    print(f"Slow-band power: {slow_power:.4f}")
    print(f"Spindle-band power: {spindle_power:.4f}")

    window = (times >= 5) & (times <= 20)
    plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(times[window], eeg[window], color="#243b53", linewidth=1.0)
    ax1.set_title("Thalamocortical EEG-like Output")
    ax1.set_ylabel("a.u.")
    ax1.grid(True, alpha=0.2)

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(times[window], cortical[window], color="#006d77", linewidth=1.0, label="Cortical pyramidal")
    ax2.plot(times[window], signals["adaptation"][window], color="#83c5be", linewidth=1.0, label="Adaptation")
    ax2.set_title("Cortical Slow Population")
    ax2.set_ylabel("a.u.")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(times[window], relay[window], color="#8d0801", linewidth=1.0, label="Thalamic relay")
    ax3.plot(times[window], reticular[window], color="#f77f00", linewidth=1.0, label="Thalamic reticular")
    ax3.set_title("Thalamic Relay/Reticular Loop")
    ax3.set_ylabel("a.u.")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.2)

    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(times[window], spindle[window], color="#6a4c93", linewidth=1.0)
    ax4.set_title("Spindle-Band Thalamic Component")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("a.u.")
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("thalamocortical_demo.png")
    print("\nSaved thalamocortical_demo.png")


if __name__ == "__main__":
    main()
