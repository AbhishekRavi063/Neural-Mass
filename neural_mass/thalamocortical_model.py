from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ThalamocorticalParameters:
    """Parameters for a compact thalamocortical sleep neural mass model."""

    dt: float = 0.001
    cortical_frequency: float = 0.8
    spindle_frequency: float = 13.0
    cortical_damping: float = 0.18
    spindle_damping: float = 0.55
    adaptation_strength: float = 0.45
    adaptation_tau: float = 1.8
    cortex_to_thalamus: float = 0.35
    thalamus_to_cortex: float = 0.28
    reticular_inhibition: float = 0.55
    relay_to_reticular: float = 0.70
    background_drive: float = 0.20
    noise_std: float = 0.015
    # Coupling scale factors
    cortical_excitation_scale: float = 18.0   # excitatory drive from cortex to relay nucleus
    reticular_inhibition_scale: float = 14.0  # inhibitory drive from reticular to relay nucleus
    spindle_feedback_gain: float = 8.0        # positive feedback within spindle oscillator
    cortical_inhibitory_weight: float = 0.35  # IPSP weight from cortical interneurons
    thalamic_relay_damping: float = 0.35      # intrinsic damping ratio of thalamic relay
    # Mixing coefficients
    reticular_down_state_mix: float = 0.4     # down-state gate contribution to reticular input
    spindle_reticular_mix: float = 0.25       # reticular contribution to spindle drive
    spindle_drive_offset: float = 0.45        # spindle oscillator activation threshold
    # EEG proxy weights
    eeg_spindle_weight: float = 0.18          # spindle contribution to EEG proxy signal
    eeg_relay_weight: float = 0.08            # relay contribution to EEG proxy signal
    # Neuromodulator (cholinergic / noradrenergic tone)
    neuromodulator_level: float = 0.0         # 0=wake/REM (high ACh), 1=deep NREM (low ACh)
    neuromodulation_strength: float = 0.35    # coupling scale factor per unit neuromodulator


class ThalamocorticalModel:
    """Compact cortex-thalamus model for sleep EEG experiments.

    Biological structure:
    - cortical pyramidal population: slow EEG / K-complex-like output
    - cortical inhibitory population: local cortical inhibition
    - thalamic relay population: thalamocortical excitation
    - thalamic reticular population: inhibitory spindle generator

    The thalamic relay/reticular loop is a spindle-band oscillator gated by the
    cortical DOWN/UP rhythm. RK4 integration is used for the deterministic part;
    noise is applied as Euler-Maruyama.
    """

    def __init__(
        self,
        parameters: ThalamocorticalParameters | None = None,
        seed: int | None = None,
    ):
        self.parameters = parameters or ThalamocorticalParameters()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(7, dtype=float)
        # Initial conditions: cortical pyramidal slightly below threshold,
        # thalamic relay slightly above zero.
        self.state[0] = -0.25
        self.state[2] = 0.05

    @staticmethod
    def sigmoid(x: NDArray | float, gain: float = 3.0, threshold: float = 0.0):
        return 1.0 / (1.0 + np.exp(-gain * (x - threshold)))

    def reset(self):
        self.state[:] = 0.0
        self.state[0] = -0.25
        self.state[2] = 0.05

    def derivatives(
        self, external_stimulus: float = 0.0, state: NDArray | None = None
    ) -> NDArray[np.float64]:
        p = self.parameters
        s = state if state is not None else self.state
        (
            cortical_pyramidal, cortical_velocity,
            thalamic_relay, thalamic_velocity,
            adaptation, spindle_x, spindle_y,
        ) = s

        # Neuromodulator scaling: low ACh/NE in NREM → stronger relay-reticular
        # loop and reduced cortical excitability (more slow waves / burst mode).
        nm_scale = 1.0 + p.neuromodulation_strength * p.neuromodulator_level
        eff_relay_to_reticular = p.relay_to_reticular * nm_scale
        eff_thalamus_to_cortex = p.thalamus_to_cortex * nm_scale
        eff_background_drive = p.background_drive * (1.0 - 0.3 * p.neuromodulator_level)

        cortical_excitation = self.sigmoid(cortical_pyramidal + eff_background_drive)
        # Cortical inhibitory interneuron activation (faster threshold than excitation)
        cortical_inhibition = self.sigmoid(cortical_excitation, gain=4.0, threshold=0.55)
        # DOWN-state gate: active when cortical pyramidal is below resting threshold
        down_state_gate = 1.0 - self.sigmoid(
            cortical_pyramidal, gain=5.0, threshold=0.05
        )
        reticular_input = (
            eff_relay_to_reticular * thalamic_relay
            + p.reticular_down_state_mix * down_state_gate
        )
        reticular_activity = self.sigmoid(reticular_input)

        slow_omega = 2.0 * np.pi * p.cortical_frequency
        spindle_omega = 2.0 * np.pi * p.spindle_frequency
        spindle_radius = spindle_x ** 2 + spindle_y ** 2
        spindle_drive = (
            down_state_gate
            + p.spindle_reticular_mix * reticular_activity
            - p.spindle_drive_offset
        )

        dy = np.zeros_like(s)
        dy[0] = cortical_velocity
        dy[1] = (
            -2.0 * p.cortical_damping * slow_omega * cortical_velocity
            - (slow_omega ** 2) * cortical_pyramidal
            - p.adaptation_strength * adaptation
            - p.cortical_inhibitory_weight * cortical_inhibition
            + eff_thalamus_to_cortex * thalamic_relay
            + external_stimulus
        )
        dy[2] = thalamic_velocity
        dy[3] = (
            -2.0 * p.thalamic_relay_damping * spindle_omega * thalamic_velocity
            - (spindle_omega ** 2) * thalamic_relay
            + p.cortical_excitation_scale * p.cortex_to_thalamus * cortical_excitation
            - p.reticular_inhibition_scale * p.reticular_inhibition * reticular_activity
            + p.spindle_feedback_gain * spindle_x
        )
        dy[4] = (-adaptation + cortical_excitation) / p.adaptation_tau
        dy[5] = (
            p.spindle_damping * spindle_drive * spindle_x
            - spindle_omega * spindle_y
            - spindle_radius * spindle_x
        )
        dy[6] = (
            spindle_omega * spindle_x
            + p.spindle_damping * spindle_drive * spindle_y
            - spindle_radius * spindle_y
        )
        return dy

    def step(self, external_stimulus: float = 0.0):
        p = self.parameters
        s0 = self.state.copy()
        dt = p.dt

        k1 = self.derivatives(external_stimulus, s0)
        k2 = self.derivatives(external_stimulus, s0 + 0.5 * dt * k1)
        k3 = self.derivatives(external_stimulus, s0 + 0.5 * dt * k2)
        k4 = self.derivatives(external_stimulus, s0 + dt * k3)

        noise = self.rng.normal(0.0, p.noise_std, size=len(s0))
        # Velocity states receive larger noise (they're driven by fast fluctuations)
        noise[[1, 3]] *= 4.0
        self.state = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + noise * np.sqrt(dt)

    def simulate(
        self, seconds: float = 30.0, stimuli: NDArray | None = None
    ) -> dict[str, NDArray[np.float64]]:
        p = self.parameters
        steps = int(seconds / p.dt)
        if steps <= 0:
            raise ValueError("seconds must produce at least one simulation step.")

        if stimuli is None:
            stimuli = np.zeros(steps, dtype=float)
        else:
            stimuli = np.asarray(stimuli, dtype=float)
            if len(stimuli) != steps:
                raise ValueError("stimuli must have one value per simulation step.")

        history = np.zeros((steps, len(self.state)), dtype=float)
        for idx in range(steps):
            self.step(float(stimuli[idx]))
            history[idx] = self.state

        cortical = history[:, 0]
        relay = history[:, 2]
        adaptation = history[:, 4]
        spindle = history[:, 5]
        reticular = self.sigmoid(p.relay_to_reticular * relay)
        eeg = cortical + p.eeg_spindle_weight * spindle + p.eeg_relay_weight * relay

        return {
            "eeg": eeg,
            "cortical_pyramidal": cortical,
            # Cortical interneuron proxy (different threshold from inhibition in derivatives)
            "cortical_interneuron": self.sigmoid(cortical, gain=4.0, threshold=0.1),
            "thalamic_relay": relay,
            "thalamic_reticular": reticular,
            "adaptation": adaptation,
            "spindle": spindle,
        }


def simulate_thalamocortical_sleep(
    seconds: float = 30.0,
    sampling_frequency: int = 200,
    seed: int | None = 7,
) -> dict[str, NDArray[np.float64]]:
    """Simulate and downsample a thalamocortical NREM-like signal."""
    dt = 0.001
    model = ThalamocorticalModel(ThalamocorticalParameters(dt=dt), seed=seed)
    raw = model.simulate(seconds=seconds)
    stride = max(1, int(round((1 / dt) / sampling_frequency)))
    return {name: values[::stride] for name, values in raw.items()}
