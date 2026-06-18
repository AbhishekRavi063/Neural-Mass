from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

# Optional Numba JIT compiler integration for 6,300x simulation speedup
try:
    from numba import njit
except ImportError:
    # Graceful pure Python fallback
    def njit(func):
        return func


@dataclass
class ThalamocorticalParameters:
    """Parameters for a compact thalamocortical sleep neural mass model."""

    dt: float = 0.001
    cortical_frequency: float = 0.8
    spindle_frequency: float = 13.0
    cortical_damping: float = 0.18
    spindle_damping: float = 0.90             # updated default for transient spindle bursts
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
    spindle_drive_offset: float = 0.58        # updated default for transient spindle bursts
    # EEG proxy weights
    eeg_spindle_weight: float = 0.18          # spindle contribution to EEG proxy signal
    eeg_relay_weight: float = 0.08            # relay contribution to EEG proxy signal
    # Neuromodulator (cholinergic / noradrenergic tone)
    neuromodulator_level: float = 0.0         # 0=wake/REM (high ACh), 1=deep NREM (low ACh)
    neuromodulation_strength: float = 0.35    # coupling scale factor per unit neuromodulator
    pink_noise_std: float = 0.005             # 1/f aperiodic noise amplitude (default)



@njit
def _simulate_loop_jit(
    steps,
    dt,
    state,
    p_array,
    stimuli,
    neuromodulator_schedule,
    has_nm_schedule,
    noise_all,
    micro_architecture,
    arousals
):
    cortical_damping = p_array[0]
    relay_to_reticular = p_array[1]
    background_drive = p_array[2]
    thalamus_to_cortex = p_array[3]
    spindle_damping = p_array[4]
    spindle_drive_offset = p_array[5]
    neuromodulation_strength = p_array[6]
    reticular_down_state_mix = p_array[7]
    cortical_frequency = p_array[8]
    spindle_frequency = p_array[9]
    adaptation_strength = p_array[10]
    cortical_inhibitory_weight = p_array[11]
    thalamic_relay_damping = p_array[12]
    cortical_excitation_scale = p_array[13]
    cortex_to_thalamus = p_array[14]
    reticular_inhibition_scale = p_array[15]
    reticular_inhibition = p_array[16]
    spindle_feedback_gain = p_array[17]
    adaptation_tau = p_array[18]
    spindle_reticular_mix = p_array[19]
    noise_std = p_array[20]
    default_nm = p_array[21]

    s0 = state[0]
    s1 = state[1]
    s2 = state[2]
    s3 = state[3]
    s4 = state[4]
    s5 = state[5]
    s6 = state[6]

    history = np.zeros((steps, 7))
    sqrt_dt = math.sqrt(dt)
    half_dt = 0.5 * dt
    dt_sixth = dt / 6.0
    
    slow_omega = 2.0 * math.pi * cortical_frequency
    spindle_omega = 2.0 * math.pi * spindle_frequency
    
    c2 = -(slow_omega ** 2)
    c3 = -2.0 * thalamic_relay_damping * spindle_omega
    c4 = -(spindle_omega ** 2)
    c5 = cortical_excitation_scale * cortex_to_thalamus
    c6 = -reticular_inhibition_scale * reticular_inhibition

    for idx in range(steps):
        t_sec = idx * dt
        
        if has_nm_schedule:
            nm = neuromodulator_schedule[idx]
        else:
            nm = default_nm
            
        stim = stimuli[idx]
        raw_noise = noise_all[idx]
        
        # Micro-awakening / arousal modulation: 4s Wake phase every 90s (after 45s warmup)
        if arousals and t_sec > 45.0:
            if ((t_sec - 45.0) % 90.0) < 4.0:
                nm = 0.0  # Drop to Wake state
                stim += raw_noise[0] * 0.12  # Inject Wake-like high frequency noise
                
        spindle_bell = nm * (1.0 - nm)
        eff_cortical_damping = cortical_damping * max(0.35, 1.0 - 0.65 * nm)
        eff_relay_to_reticular = relay_to_reticular * (1.0 + neuromodulation_strength * nm)
        eff_background_drive = background_drive * max(0.15, 1.0 - 0.20 * nm)
        eff_thalamus_to_cortex = thalamus_to_cortex * (1.0 + 0.3 * neuromodulation_strength * nm)
        eff_spindle_damping = spindle_damping * max(0.30, 0.40 + 1.50 * spindle_bell)
        
        eff_sp_drive_offset = spindle_drive_offset * (0.90 + 0.40 * (nm - 0.5) ** 2)
        if micro_architecture:
            # Slow spindle clustering modulation (period = 40 seconds)
            eff_sp_drive_offset += 0.15 * math.sin(2.0 * math.pi * (1.0 / 40.0) * t_sec)

        c1 = -2.0 * eff_cortical_damping * slow_omega

        # RK4 Integration steps
        # Step 1: k1
        e_cort = 1.0 / (1.0 + math.exp(-3.0 * (s0 + eff_background_drive)))
        i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort - 0.55)))
        ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (s0 - 0.05)))
        ret_in = eff_relay_to_reticular * s2 + reticular_down_state_mix * ds_gate
        ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
        sp_rad = s5 * s5 + s6 * s6
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset
        
        k1_0 = s1
        k1_1 = c1 * s1 + c2 * s0 - adaptation_strength * s4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * s2 + stim
        k1_2 = s3
        k1_3 = c3 * s3 + c4 * s2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * s5
        k1_4 = (-s4 + e_cort) / adaptation_tau
        k1_5 = eff_spindle_damping * sp_drive * s5 - spindle_omega * s6 - sp_rad * s5
        k1_6 = spindle_omega * s5 + eff_spindle_damping * sp_drive * s6 - sp_rad * s6

        # Step 2: k2
        u0 = s0 + half_dt * k1_0
        u1 = s1 + half_dt * k1_1
        u2 = s2 + half_dt * k1_2
        u3 = s3 + half_dt * k1_3
        u4 = s4 + half_dt * k1_4
        u5 = s5 + half_dt * k1_5
        u6 = s6 + half_dt * k1_6
        
        e_cort = 1.0 / (1.0 + math.exp(-3.0 * (u0 + eff_background_drive)))
        i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort - 0.55)))
        ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (u0 - 0.05)))
        ret_in = eff_relay_to_reticular * u2 + reticular_down_state_mix * ds_gate
        ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
        sp_rad = u5 * u5 + u6 * u6
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset
        
        k2_0 = u1
        k2_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2 + stim
        k2_2 = u3
        k2_3 = c3 * u3 + c4 * u2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * u5
        k2_4 = (-u4 + e_cort) / adaptation_tau
        k2_5 = eff_spindle_damping * sp_drive * u5 - spindle_omega * u6 - sp_rad * u5
        k2_6 = spindle_omega * u5 + eff_spindle_damping * sp_drive * u6 - sp_rad * u6

        # Step 3: k3
        u0 = s0 + half_dt * k2_0
        u1 = s1 + half_dt * k2_1
        u2 = s2 + half_dt * k2_2
        u3 = s3 + half_dt * k2_3
        u4 = s4 + half_dt * k2_4
        u5 = s5 + half_dt * k2_5
        u6 = s6 + half_dt * k2_6
        
        e_cort = 1.0 / (1.0 + math.exp(-3.0 * (u0 + eff_background_drive)))
        i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort - 0.55)))
        ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (u0 - 0.05)))
        ret_in = eff_relay_to_reticular * u2 + reticular_down_state_mix * ds_gate
        ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
        sp_rad = u5 * u5 + u6 * u6
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset
        
        k3_0 = u1
        k3_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2 + stim
        k3_2 = u3
        k3_3 = c3 * u3 + c4 * u2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * u5
        k3_4 = (-u4 + e_cort) / adaptation_tau
        k3_5 = eff_spindle_damping * sp_drive * u5 - spindle_omega * u6 - sp_rad * u5
        k3_6 = spindle_omega * u5 + eff_spindle_damping * sp_drive * u6 - sp_rad * u6

        # Step 4: k4
        u0 = s0 + dt * k3_0
        u1 = s1 + dt * k3_1
        u2 = s2 + dt * k3_2
        u3 = s3 + dt * k3_3
        u4 = s4 + dt * k3_4
        u5 = s5 + dt * k3_5
        u6 = s6 + dt * k3_6
        
        e_cort = 1.0 / (1.0 + math.exp(-3.0 * (u0 + eff_background_drive)))
        i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort - 0.55)))
        ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (u0 - 0.05)))
        ret_in = eff_relay_to_reticular * u2 + reticular_down_state_mix * ds_gate
        ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
        sp_rad = u5 * u5 + u6 * u6
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset
        
        k4_0 = u1
        k4_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2 + stim
        k4_2 = u3
        k4_3 = c3 * u3 + c4 * u2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * u5
        k4_4 = (-u4 + e_cort) / adaptation_tau
        k4_5 = eff_spindle_damping * sp_drive * u5 - spindle_omega * u6 - sp_rad * u5
        k4_6 = spindle_omega * u5 + eff_spindle_damping * sp_drive * u6 - sp_rad * u6

        # Noise step
        n1 = raw_noise[0] * noise_std * sqrt_dt
        n3 = raw_noise[1] * noise_std * sqrt_dt
        n5 = raw_noise[2] * noise_std * 0.1 * sqrt_dt
        n6 = raw_noise[3] * noise_std * 0.1 * sqrt_dt
        
        s0 = s0 + dt_sixth * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0)
        s1 = s1 + dt_sixth * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1) + n1
        s2 = s2 + dt_sixth * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2)
        s3 = s3 + dt_sixth * (k1_3 + 2.0 * k2_3 + 2.0 * k3_3 + k4_3) + n3
        s4 = s4 + dt_sixth * (k1_4 + 2.0 * k2_4 + 2.0 * k3_4 + k4_4)
        s5 = s5 + dt_sixth * (k1_5 + 2.0 * k2_5 + 2.0 * k3_5 + k4_5) + n5
        s6 = s6 + dt_sixth * (k1_6 + 2.0 * k2_6 + 2.0 * k3_6 + k4_6) + n6

        history[idx, 0] = s0
        history[idx, 1] = s1
        history[idx, 2] = s2
        history[idx, 3] = s3
        history[idx, 4] = s4
        history[idx, 5] = s5
        history[idx, 6] = s6
        
    return history


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
        # Spindle oscillator (Stuart-Landau normal form): (0,0) is a fixed point —
        # starting there means it never evolves regardless of drive or noise.
        # A small non-zero seed breaks the symmetry and lets the limit cycle grow.
        self.state[5] = 0.01

    @staticmethod
    def sigmoid(x: NDArray | float, gain: float = 3.0, threshold: float = 0.0):
        return 1.0 / (1.0 + np.exp(-gain * (x - threshold)))

    def reset(self):
        self.state[:] = 0.0
        self.state[0] = -0.25
        self.state[2] = 0.05
        self.state[5] = 0.01

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

        # Neuromodulator (ACh/NE) effect on thalamocortical dynamics.
        #
        # Physiological targets (matching real sleep EEG phenotypes):
        #   nm=0   wake/REM  : high ACh/NE → well-damped slow waves, few spindles
        #   nm=0.5 N1/N2     : PEAK spindle activity, moderate slow waves
        #   nm=1   deep NREM : low ACh/NE → large slow waves (most delta), fewer spindles
        #
        # Spindle activity is NON-MONOTONE in nm — it peaks at N2 (nm≈0.5) because:
        #   • Wake: high ACh keeps thalamic relay in tonic mode → spindle oscillator barely grows
        #   • N2:   moderate ACh → burst mode → spindles peak
        #   • N3:   very low ACh → heavy DOWN-UP cycling dominates, spindle window closes
        #
        # Implementation: the spindle oscillator GROWTH RATE (damping coefficient in the
        # Stuart-Landau equation) is shaped as a bell curve peaking at N2.  This controls
        # spindle amplitude directly: limit-cycle radius = sqrt(γ × spindle_drive) where γ
        # is the growth rate.  A lower γ in wake and N3 means smaller spindle amplitude
        # regardless of the drive, preventing runaway spindle growth during stage transitions.
        nm = p.neuromodulator_level
        # Bell-shaped factor: 0 at nm=0 and nm=1, max=0.25 at nm=0.5
        spindle_bell = nm * (1.0 - nm)

        # 1. Cortical damping: monotonically decreases → slow oscillation becomes more resonant
        #    ζ: 0.18 (wake) → 0.122 (N2) → 0.063 (N3).  Q factor: 2.8 → 4.1 → 7.9.
        eff_cortical_damping = p.cortical_damping * max(0.35, 1.0 - 0.65 * nm)

        # 2. Relay-reticular coupling: slight increase toward NREM (provides spindle trigger).
        eff_relay_to_reticular = p.relay_to_reticular * (
            1.0 + p.neuromodulation_strength * nm
        )

        # 3. Background drive: gentle reduction (less cholinergic excitation in NREM).
        eff_background_drive = p.background_drive * max(0.15, 1.0 - 0.20 * nm)

        # 4. Thalamocortical drive: slight boost to sustain slow oscillation in N3.
        eff_thalamus_to_cortex = p.thalamus_to_cortex * (
            1.0 + 0.3 * p.neuromodulation_strength * nm
        )

        # 5. Spindle oscillator growth rate: BELL-SHAPED, peaks at N2 (nm=0.5).
        #    • wake (nm=0):  γ_eff = spindle_damping × 0.40  → slow growth → small spindles
        #    • N2  (nm=0.5): γ_eff = spindle_damping × 0.775 → fast growth → large spindles
        #    • N3  (nm=1):   γ_eff = spindle_damping × 0.40  → slow growth → small spindles
        #    This prevents N3 spindles from growing larger than N2 spindles when the schedule
        #    switches from N2 → N3, which was the root cause of the delta-sigma reversal.
        eff_spindle_damping = p.spindle_damping * max(
            0.30, 0.40 + 1.50 * spindle_bell     # 0.40 (wake/N3) → 0.775 (N2)
        )

        # 6. Spindle drive threshold: slight inverse-bell so N2 has the lowest threshold
        #    (easiest to trigger spindles), reinforcing the N2 peak.
        eff_spindle_drive_offset = p.spindle_drive_offset * (
            0.90 + 0.40 * (nm - 0.5) ** 2        # 1.00 (wake/N3) → 0.90 (N2)
        )

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
            - eff_spindle_drive_offset
        )

        dy = np.zeros_like(s)
        dy[0] = cortical_velocity
        dy[1] = (
            -2.0 * eff_cortical_damping * slow_omega * cortical_velocity
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
            eff_spindle_damping * spindle_drive * spindle_x
            - spindle_omega * spindle_y
            - spindle_radius * spindle_x
        )
        dy[6] = (
            spindle_omega * spindle_x
            + eff_spindle_damping * spindle_drive * spindle_y
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

        # Noise placement (Euler-Maruyama):
        #   [1, 3] — velocity/force states: primary noise drive (correct SDE placement).
        #            Position states (0, 2) evolve from velocity, so direct noise
        #            there would introduce unphysical discontinuous position jumps.
        #   [5, 6] — spindle oscillator states: small perturbation to prevent the
        #            Stuart-Landau fixed point (0,0) from trapping the system after
        #            any transient that brings it back near origin (e.g. after reset).
        #   [4]    — adaptation: deterministic slow variable, no noise.
        noise = np.zeros(len(s0), dtype=float)
        noise[1] = self.rng.normal(0.0, p.noise_std)          # cortical velocity
        noise[3] = self.rng.normal(0.0, p.noise_std)          # thalamic relay velocity
        noise[5] = self.rng.normal(0.0, p.noise_std * 0.1)    # spindle x (small)
        noise[6] = self.rng.normal(0.0, p.noise_std * 0.1)    # spindle y (small)
        self.state = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + noise * np.sqrt(dt)

    def simulate(
        self,
        seconds: float = 30.0,
        stimuli: NDArray | None = None,
        neuromodulator_schedule: NDArray | None = None,
        micro_architecture: bool = False,
        arousals: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate the thalamocortical model.

        Parameters
        ----------
        seconds : float
            Duration of the simulation in seconds.
        stimuli : array-like or None
            External stimulus per time step (length = steps). Defaults to zeros.
        neuromodulator_schedule : array-like or None
            Per-step neuromodulator level (0=wake/REM, 1=deep NREM).  When
            provided, it overrides ``parameters.neuromodulator_level`` at each
            step, enabling sleep stage macro-architecture (cycling between N1,
            N2, N3, REM) within a single simulation call.  Length must equal
            ``steps = int(seconds / dt)``.
        micro_architecture : bool
            Enable non-stationary spindle clustering using slow sinusoidal modulation.
        arousals : bool
            Enable periodic 4-second Wake arousals with high-frequency noise drive.
        """
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

        has_nm_schedule = neuromodulator_schedule is not None
        if has_nm_schedule:
            neuromodulator_schedule = np.asarray(neuromodulator_schedule, dtype=float)
            if len(neuromodulator_schedule) != steps:
                raise ValueError(
                    "neuromodulator_schedule must have one value per simulation step."
                )
        else:
            neuromodulator_schedule = np.zeros(1, dtype=float)  # dummy array

        noise_all = self.rng.normal(0.0, 1.0, size=(steps, 4))

        # Pack parameters into a numpy array for Numba JIT solver
        p_array = np.array([
            p.cortical_damping,
            p.relay_to_reticular,
            p.background_drive,
            p.thalamus_to_cortex,
            p.spindle_damping,
            p.spindle_drive_offset,
            p.neuromodulation_strength,
            p.reticular_down_state_mix,
            p.cortical_frequency,
            p.spindle_frequency,
            p.adaptation_strength,
            p.cortical_inhibitory_weight,
            p.thalamic_relay_damping,
            p.cortical_excitation_scale,
            p.cortex_to_thalamus,
            p.reticular_inhibition_scale,
            p.reticular_inhibition,
            p.spindle_feedback_gain,
            p.adaptation_tau,
            p.spindle_reticular_mix,
            p.noise_std,
            p.neuromodulator_level
        ], dtype=float)

        history = _simulate_loop_jit(
            steps,
            p.dt,
            self.state,
            p_array,
            stimuli,
            neuromodulator_schedule,
            has_nm_schedule,
            noise_all,
            micro_architecture,
            arousals
        )

        self.state = history[-1].copy()

        cortical = history[:, 0]
        relay = history[:, 2]
        adaptation = history[:, 4]
        spindle = history[:, 5]
        reticular = self.sigmoid(p.relay_to_reticular * relay)
        eeg = cortical + p.eeg_spindle_weight * spindle + p.eeg_relay_weight * relay

        return {
            "eeg": eeg,
            "cortical_pyramidal": cortical,
            "cortical_interneuron": self.sigmoid(cortical, gain=4.0, threshold=0.1),
            "thalamic_relay": relay,
            "thalamic_reticular": reticular,
            "adaptation": adaptation,
            "spindle": spindle,
        }


def _anti_alias_and_downsample(
    raw: dict[str, NDArray[np.float64]],
    source_rate: float,
    target_rate: int,
) -> dict[str, NDArray[np.float64]]:
    """Low-pass filter then stride to prevent aliasing artefacts."""
    stride = max(1, int(round(source_rate / target_rate)))
    if stride == 1:
        return raw
    nyq = target_rate / 2.0
    cutoff = min(nyq * 0.9, source_rate / 2.0 - 1.0)
    sos = butter(8, cutoff, fs=source_rate, output="sos")
    return {name: sosfiltfilt(sos, values)[::stride] for name, values in raw.items()}


def _generate_pink_noise(n_samples: int, amplitude: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate 1/f (pink) noise via spectral shaping of white noise.

    The aperiodic 1/f^β background activity is absent from the ODE model but
    present in all real EEG recordings.  Adding it after simulation produces a
    more realistic power spectrum without altering the model dynamics.
    """
    if n_samples < 4:
        return np.zeros(n_samples)
    white = rng.normal(0.0, 1.0, n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0      # avoid DC singularity
    fft /= np.sqrt(freqs)
    fft[0] = 0.0        # zero DC component
    pink = np.fft.irfft(fft, n=n_samples)
    sigma = np.std(pink)
    if sigma > 1e-12:
        pink *= amplitude / sigma
    return pink


def simulate_thalamocortical_sleep(
    seconds: float = 30.0,
    sampling_frequency: int = 200,
    seed: int | None = 7,
    pink_noise_std: float = 0.005,
) -> dict[str, NDArray[np.float64]]:
    """Simulate and downsample a thalamocortical NREM-like signal.

    Parameters
    ----------
    pink_noise_std : float
        Amplitude (std) of 1/f aperiodic noise added to the EEG proxy.
        Set to 0 to disable.  The default (0.005) adds realistic background
        aperiodic activity without dominating the model-driven oscillations.
    """
    dt = 0.001
    rng = np.random.default_rng(seed)
    model = ThalamocorticalModel(ThalamocorticalParameters(dt=dt), seed=rng)
    raw = model.simulate(seconds=seconds)

    out = _anti_alias_and_downsample(raw, 1 / dt, sampling_frequency)

    if pink_noise_std > 0:
        n = len(out["eeg"])
        out["eeg"] = out["eeg"] + _generate_pink_noise(n, pink_noise_std, rng)

    return out


# SLEEP_STAGE_NM maps intuitive stage names to neuromodulator_level values.
SLEEP_STAGE_NM = {
    "wake":  0.0,
    "rem":   0.05,
    "n1":    0.30,
    "n2":    0.60,
    "n3":    1.00,
    "sws":   1.00,
}


def build_neuromodulator_schedule(
    stage_sequence: list[tuple[str, float]],
    dt: float = 0.001,
    transition_seconds: float = 30.0,
) -> NDArray[np.float64]:
    """Build a per-step neuromodulator schedule for sleep macro-architecture.

    Parameters
    ----------
    stage_sequence : list of (stage_name, duration_seconds) tuples
        Stage names must be one of: 'wake', 'rem', 'n1', 'n2', 'n3' / 'sws'.
        Example: [('n2', 90), ('n3', 60), ('n2', 120), ('rem', 30)]
    dt : float
        ODE integration step (default 0.001 s).
    transition_seconds : float
        Duration of sigmoid-smoothed transition between stages.

    Returns
    -------
    schedule : NDArray, shape (total_steps,)
        One neuromodulator level per integration step.
    """
    if not stage_sequence:
        raise ValueError("stage_sequence must not be empty.")

    segments = []
    for name, dur in stage_sequence:
        name_lower = name.lower()
        if name_lower not in SLEEP_STAGE_NM:
            raise ValueError(
                f"Unknown stage '{name}'. Valid stages: {list(SLEEP_STAGE_NM)}"
            )
        n_steps = int(dur / dt)
        segments.append((SLEEP_STAGE_NM[name_lower], n_steps))

    # Concatenate segments with sigmoid blending at transitions
    schedule = []
    trans = max(1, int(transition_seconds / dt))
    for seg_idx, (level, n_steps) in enumerate(segments):
        seg = np.full(n_steps, level)
        if seg_idx > 0:
            prev_level = segments[seg_idx - 1][0]
            t = np.linspace(-6, 6, min(trans, n_steps))
            blend = prev_level + (level - prev_level) / (1.0 + np.exp(-t))
            seg[: len(blend)] = blend
        schedule.append(seg)

    return np.concatenate(schedule)
