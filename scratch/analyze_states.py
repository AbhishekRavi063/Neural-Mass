import numpy as np
from neural_mass import ThalamocorticalSleepModel

for nm in [0.0, 0.3, 0.6, 1.0]:
    model = ThalamocorticalSleepModel(neuromodulator_level=nm, seed=42)
    signals = model.simulate(seconds=60.0, sampling_frequency=200)
    eeg = signals["eeg"]
    cortical = signals["cortical_pyramidal"]
    spindle = signals["spindle"]
    relay = signals["thalamic_relay"]
    
    print(f"--- NM = {nm} ---")
    print(f"  EEG Std:      {np.std(eeg):.6f}")
    print(f"  Cortical Std: {np.std(cortical):.6f}")
    print(f"  Spindle Std:  {np.std(spindle):.6f}")
    print(f"  Relay Std:    {np.std(relay):.6f}")
