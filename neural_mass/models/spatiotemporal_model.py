from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray
from neural_mass.models.thalamocortical_model import ThalamocorticalParameters, _anti_alias_and_downsample

# Try importing Numba
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func


@njit
def _simulate_spatiotemporal_loop_jit(
    steps: int,
    n_nodes: int,
    dt: float,
    state: NDArray[np.float64],  # shape (n_nodes, 7)
    p_array: NDArray[np.float64],
    W: NDArray[np.float64],      # shape (n_nodes, n_nodes) coupling matrix
    noise_all: NDArray[np.float64],  # shape (steps, n_nodes, 4)
    lateral_coupling_strength: float,
    pacemaker_strength: float,
    closed_loop: bool,
    tau_accum: float,
    tau_dissip: float,
    initial_sleep_pressure: float,
):
    # Unpack parameters from p_array
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

    history = np.zeros((steps, n_nodes, 7))
    eeg_history = np.zeros((steps, n_nodes))
    
    # Process S variables
    sleep_pressure = initial_sleep_pressure
    s_history = np.zeros((steps, n_nodes))
    swa_envelope = np.zeros(n_nodes)
    tau_swa = 2.0  # seconds

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

    # Working state buffers
    s_curr = state.copy()
    dy1 = np.zeros((n_nodes, 7))
    dy2 = np.zeros((n_nodes, 7))
    dy3 = np.zeros((n_nodes, 7))
    dy4 = np.zeros((n_nodes, 7))

    for idx in range(steps):
        t_sec = idx * dt

        # 1. Closed-loop Sleep Stage Dynamics (Process S)
        if closed_loop:
            # SWA envelope: low-pass filter of absolute cortical velocity (baseline-independent)
            for i in range(n_nodes):
                d_swa = (abs(s_curr[i, 1]) - swa_envelope[i]) / (tau_swa)
                swa_envelope[i] += d_swa * dt
            
            # Global Process S update
            avg_envelope = 0.0
            for i in range(n_nodes):
                avg_envelope += swa_envelope[i]
            avg_envelope /= n_nodes
            
            # SWA threshold 0.05
            if avg_envelope > 0.05:
                d_S = -sleep_pressure / tau_dissip
            else:
                d_S = (1.0 - sleep_pressure) / tau_accum
            
            sleep_pressure += d_S * dt
            
            # Bound S to [0, 1]
            if sleep_pressure < 0.0:
                sleep_pressure = 0.0
            elif sleep_pressure > 1.0:
                sleep_pressure = 1.0
                
            nm = sleep_pressure
        else:
            nm = default_nm

        # Modulations based on neuromodulator level
        spindle_bell = nm * (1.0 - nm)
        eff_cortical_damping = cortical_damping * max(0.35, 1.0 - 0.65 * nm)
        eff_relay_to_reticular = relay_to_reticular * (1.0 + neuromodulation_strength * nm)
        eff_thalamus_to_cortex = thalamus_to_cortex * (1.0 + 0.3 * neuromodulation_strength * nm)
        eff_spindle_damping = spindle_damping * max(0.30, 0.40 + 1.50 * spindle_bell)
        eff_sp_drive_offset = spindle_drive_offset * (0.90 + 0.40 * (nm - 0.5) ** 2)

        c1 = -2.0 * eff_cortical_damping * slow_omega

        # --- RK4 derivatives helper ---
        # We need to evaluate derivatives for all nodes. Since they are coupled via W,
        # we compute cortical_excitation for all nodes first.
        
        # Helper function for derivatives inside JIT
        # k1
        e_cort = np.zeros(n_nodes)
        for i in range(n_nodes):
            # Pacemaker: node 0 has higher background drive
            bg_drive = background_drive
            if i == 0:
                bg_drive += pacemaker_strength
            e_cort[i] = 1.0 / (1.0 + math.exp(-3.0 * (s_curr[i, 0] + bg_drive)))

        # Lateral coupling input to cortex
        lateral_input = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                lateral_input[i] += W[i, j] * e_cort[j]
            lateral_input[i] *= lateral_coupling_strength

        for i in range(n_nodes):
            i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort[i] - 0.55)))
            ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (s_curr[i, 0] - 0.05)))
            ret_in = eff_relay_to_reticular * s_curr[i, 2] + reticular_down_state_mix * ds_gate
            ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
            sp_rad = s_curr[i, 5] * s_curr[i, 5] + s_curr[i, 6] * s_curr[i, 6]
            sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset

            dy1[i, 0] = s_curr[i, 1]
            dy1[i, 1] = c1 * s_curr[i, 1] + c2 * s_curr[i, 0] - adaptation_strength * s_curr[i, 4] - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * s_curr[i, 2] + lateral_input[i]
            dy1[i, 2] = s_curr[i, 3]
            dy1[i, 3] = c3 * s_curr[i, 3] + c4 * s_curr[i, 2] + c5 * e_cort[i] + c6 * ret_act + spindle_feedback_gain * s_curr[i, 5]
            dy1[i, 4] = (-s_curr[i, 4] + e_cort[i]) / adaptation_tau
            dy1[i, 5] = eff_spindle_damping * sp_drive * s_curr[i, 5] - spindle_omega * s_curr[i, 6] - sp_rad * s_curr[i, 5]
            dy1[i, 6] = spindle_omega * s_curr[i, 5] + eff_spindle_damping * sp_drive * s_curr[i, 6] - sp_rad * s_curr[i, 6]

        # k2
        u = np.zeros((n_nodes, 7))
        for i in range(n_nodes):
            for m in range(7):
                u[i, m] = s_curr[i, m] + half_dt * dy1[i, m]

        e_cort = np.zeros(n_nodes)
        for i in range(n_nodes):
            bg_drive = background_drive
            if i == 0:
                bg_drive += pacemaker_strength
            e_cort[i] = 1.0 / (1.0 + math.exp(-3.0 * (u[i, 0] + bg_drive)))

        lateral_input = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                lateral_input[i] += W[i, j] * e_cort[j]
            lateral_input[i] *= lateral_coupling_strength

        for i in range(n_nodes):
            i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort[i] - 0.55)))
            ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (u[i, 0] - 0.05)))
            ret_in = eff_relay_to_reticular * u[i, 2] + reticular_down_state_mix * ds_gate
            ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
            sp_rad = u[i, 5] * u[i, 5] + u[i, 6] * u[i, 6]
            sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset

            dy2[i, 0] = u[i, 1]
            dy2[i, 1] = c1 * u[i, 1] + c2 * u[i, 0] - adaptation_strength * u[i, 4] - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u[i, 2] + lateral_input[i]
            dy2[i, 2] = u[i, 3]
            dy2[i, 3] = c3 * u[i, 3] + c4 * u[i, 2] + c5 * e_cort[i] + c6 * ret_act + spindle_feedback_gain * u[i, 5]
            dy2[i, 4] = (-u[i, 4] + e_cort[i]) / adaptation_tau
            dy2[i, 5] = eff_spindle_damping * sp_drive * u[i, 5] - spindle_omega * u[i, 6] - sp_rad * u[i, 5]
            dy2[i, 6] = spindle_omega * u[i, 5] + eff_spindle_damping * sp_drive * u[i, 6] - sp_rad * u[i, 6]

        # k3
        for i in range(n_nodes):
            for m in range(7):
                u[i, m] = s_curr[i, m] + half_dt * dy2[i, m]

        e_cort = np.zeros(n_nodes)
        for i in range(n_nodes):
            bg_drive = background_drive
            if i == 0:
                bg_drive += pacemaker_strength
            e_cort[i] = 1.0 / (1.0 + math.exp(-3.0 * (u[i, 0] + bg_drive)))

        lateral_input = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                lateral_input[i] += W[i, j] * e_cort[j]
            lateral_input[i] *= lateral_coupling_strength

        for i in range(n_nodes):
            i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort[i] - 0.55)))
            ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (u[i, 0] - 0.05)))
            ret_in = eff_relay_to_reticular * u[i, 2] + reticular_down_state_mix * ds_gate
            ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
            sp_rad = u[i, 5] * u[i, 5] + u[i, 6] * u[i, 6]
            sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset

            dy3[i, 0] = u[i, 1]
            dy3[i, 1] = c1 * u[i, 1] + c2 * u[i, 0] - adaptation_strength * u[i, 4] - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u[i, 2] + lateral_input[i]
            dy3[i, 2] = u[i, 3]
            dy3[i, 3] = c3 * u[i, 3] + c4 * u[i, 2] + c5 * e_cort[i] + c6 * ret_act + spindle_feedback_gain * u[i, 5]
            dy3[i, 4] = (-u[i, 4] + e_cort[i]) / adaptation_tau
            dy3[i, 5] = eff_spindle_damping * sp_drive * u[i, 5] - spindle_omega * u[i, 6] - sp_rad * u[i, 5]
            dy3[i, 6] = spindle_omega * u[i, 5] + eff_spindle_damping * sp_drive * u[i, 6] - sp_rad * u[i, 6]

        # k4
        for i in range(n_nodes):
            for m in range(7):
                u[i, m] = s_curr[i, m] + dt * dy3[i, m]

        e_cort = np.zeros(n_nodes)
        for i in range(n_nodes):
            bg_drive = background_drive
            if i == 0:
                bg_drive += pacemaker_strength
            e_cort[i] = 1.0 / (1.0 + math.exp(-3.0 * (u[i, 0] + bg_drive)))

        lateral_input = np.zeros(n_nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                lateral_input[i] += W[i, j] * e_cort[j]
            lateral_input[i] *= lateral_coupling_strength

        for i in range(n_nodes):
            i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort[i] - 0.55)))
            ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (u[i, 0] - 0.05)))
            ret_in = eff_relay_to_reticular * u[i, 2] + reticular_down_state_mix * ds_gate
            ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
            sp_rad = u[i, 5] * u[i, 5] + u[i, 6] * u[i, 6]
            sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_sp_drive_offset

            dy4[i, 0] = u[i, 1]
            dy4[i, 1] = c1 * u[i, 1] + c2 * u[i, 0] - adaptation_strength * u[i, 4] - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u[i, 2] + lateral_input[i]
            dy4[i, 2] = u[i, 3]
            dy4[i, 3] = c3 * u[i, 3] + c4 * u[i, 2] + c5 * e_cort[i] + c6 * ret_act + spindle_feedback_gain * u[i, 5]
            dy4[i, 4] = (-u[i, 4] + e_cort[i]) / adaptation_tau
            dy4[i, 5] = eff_spindle_damping * sp_drive * u[i, 5] - spindle_omega * u[i, 6] - sp_rad * u[i, 5]
            dy4[i, 6] = spindle_omega * u[i, 5] + eff_spindle_damping * sp_drive * u[i, 6] - sp_rad * u[i, 6]

        # Update states and add noise (Euler-Maruyama)
        for i in range(n_nodes):
            n1 = noise_all[idx, i, 0] * noise_std * sqrt_dt
            n3 = noise_all[idx, i, 1] * noise_std * sqrt_dt
            n5 = noise_all[idx, i, 2] * noise_std * 0.1 * sqrt_dt
            n6 = noise_all[idx, i, 3] * noise_std * 0.1 * sqrt_dt

            s_curr[i, 0] += dt_sixth * (dy1[i, 0] + 2.0 * dy2[i, 0] + 2.0 * dy3[i, 0] + dy4[i, 0])
            s_curr[i, 1] += dt_sixth * (dy1[i, 1] + 2.0 * dy2[i, 1] + 2.0 * dy3[i, 1] + dy4[i, 1]) + n1
            s_curr[i, 2] += dt_sixth * (dy1[i, 2] + 2.0 * dy2[i, 2] + 2.0 * dy3[i, 2] + dy4[i, 2])
            s_curr[i, 3] += dt_sixth * (dy1[i, 3] + 2.0 * dy2[i, 3] + 2.0 * dy3[i, 3] + dy4[i, 3]) + n3
            s_curr[i, 4] += dt_sixth * (dy1[i, 4] + 2.0 * dy2[i, 4] + 2.0 * dy3[i, 4] + dy4[i, 4])
            s_curr[i, 5] += dt_sixth * (dy1[i, 5] + 2.0 * dy2[i, 5] + 2.0 * dy3[i, 5] + dy4[i, 5]) + n5
            s_curr[i, 6] += dt_sixth * (dy1[i, 6] + 2.0 * dy2[i, 6] + 2.0 * dy3[i, 6] + dy4[i, 6]) + n6

            # Store history
            for m in range(7):
                history[idx, i, m] = s_curr[i, m]

            # Single channel EEG proxy output for this node
            eeg_history[idx, i] = s_curr[i, 0] + p_array[22] * s_curr[i, 5] + p_array[23] * s_curr[i, 2]
            
            # Store sleep pressure history
            s_history[idx, i] = sleep_pressure

    return history, eeg_history, s_history


class SpatiotemporalThalamocorticalModel:
    """Spatiotemporal cortical-thalamic lattice model for traveling sleep wave simulations.

    Models a 1D chain of N coupled nodes where neighboring nodes are connected via
    lateral intracortical excitation. Pacemaking activity at node 0 naturally triggers
    anterior-to-posterior slow-wave sleep traveling waves. Also incorporates Process S
    sleep pressure closed-loop feedback.
    """

    def __init__(
        self,
        n_nodes: int = 8,
        parameters: ThalamocorticalParameters | None = None,
        lateral_coupling_strength: float = 1.2,
        spatial_spread: float = 1.0,
        pacemaker_strength: float = 0.25,
        seed: int | None = None,
    ):
        self.n_nodes = n_nodes
        self.parameters = parameters or ThalamocorticalParameters()
        self.lateral_coupling_strength = lateral_coupling_strength
        self.spatial_spread = spatial_spread
        self.pacemaker_strength = pacemaker_strength
        self.rng = np.random.default_rng(seed)

        # Initialise state: shape (n_nodes, 7)
        self.state = np.zeros((self.n_nodes, 7), dtype=float)
        self.state[:, 0] = -0.25   # cortical pyramidal slightly below resting
        self.state[:, 2] = 0.05    # thalamic relay slightly above zero
        self.state[:, 5] = 0.01    # break spindle Stuart-Landau symmetry

        # Precompute Gaussian spatial coupling matrix W (diagonal = 0, row-normalised)
        self.W = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.W[i, j] = np.exp(-((i - j) ** 2) / (2.0 * (self.spatial_spread ** 2)))
            row_sum = self.W[i, :].sum()
            if row_sum > 0:
                self.W[i, :] /= row_sum

    def simulate(
        self,
        seconds: float = 30.0,
        sampling_frequency: int = 200,
        closed_loop: bool = False,
        tau_accum: float = 15.0,
        tau_dissip: float = 20.0,
        initial_sleep_pressure: float = 0.60,
    ) -> dict[str, NDArray[np.float64]]:
        """Simulate the spatiotemporal lattice model."""
        p = self.parameters
        steps = int(seconds / p.dt)
        if steps <= 0:
            raise ValueError("seconds must produce at least one step.")

        # Generate noise inputs
        noise_all = self.rng.normal(0.0, 1.0, size=(steps, self.n_nodes, 4))

        # Pack parameters into p_array for JIT loop
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
            p.neuromodulator_level,
            p.eeg_spindle_weight,
            p.eeg_relay_weight
        ], dtype=float)

        history, eeg_history, s_history = _simulate_spatiotemporal_loop_jit(
            steps,
            self.n_nodes,
            p.dt,
            self.state,
            p_array,
            self.W,
            noise_all,
            self.lateral_coupling_strength,
            self.pacemaker_strength,
            closed_loop,
            tau_accum,
            tau_dissip,
            initial_sleep_pressure
        )

        # Store last state back
        self.state = history[-1].copy()

        # Build output dictionary
        raw_out = {
            "cortical_pyramidal": history[:, :, 0],
            "thalamic_relay": history[:, :, 2],
            "spindle": history[:, :, 5],
            "eeg_nodes": eeg_history,
            "sleep_pressure": s_history[:, 0]  # save sleep pressure timeline
        }

        # Apply lowpass filtering and downsampling to each channel
        downsampled = {}
        stride = max(1, int(round((1 / p.dt) / sampling_frequency)))
        
        # We downsample each node signal individually
        # Filter raw outputs using a shared filter function
        for key, value in raw_out.items():
            if key == "sleep_pressure":
                downsampled[key] = value[::stride]
            else:
                # Shape is (steps, n_nodes)
                n_ch = value.shape[1]
                temp_dict = {str(ch): value[:, ch] for ch in range(n_ch)}
                ds_temp = _anti_alias_and_downsample(temp_dict, 1 / p.dt, sampling_frequency)
                
                # Reassemble matrix of shape (ds_steps, n_nodes)
                ch_keys = sorted(list(ds_temp.keys()), key=int)
                downsampled[key] = np.column_stack([ds_temp[k] for k in ch_keys])

        # 3. PHYSICAL 3D FORWARD PROJECTION FROM LATTICE
        # Project 1D chain of 8 nodes to frontal (Fz), central (Cz), and parietal (Pz) scalp electrodes.
        # Electrode locations on unit sphere R=1
        r_electrodes = {
            "eeg_fz": np.array([0.50, 0.0, 0.866]),
            "eeg_cz": np.array([0.0, 0.0, 1.0]),
            "eeg_pz": np.array([-0.50, 0.0, 0.866]),
        }

        # Node coordinates along sagittal midline (interpolating from x=0.6 to x=-0.6)
        x_coords = np.linspace(0.60, -0.60, self.n_nodes)
        node_positions = []
        for x in x_coords:
            # Place on the upper hemisphere surface (z = sqrt(1 - x^2))
            z = math.sqrt(max(0.0, 1.0 - x * x))
            node_positions.append(np.array([x, 0.0, z]))

        # Dipole orientations pointing radially outward
        dipole_orientations = [pos / np.linalg.norm(pos) for pos in node_positions]

        # Calculate lead field coefficients matrix (3 electrodes x N nodes)
        lead_field_c = np.zeros((3, self.n_nodes))
        lead_field_t = np.zeros((3, self.n_nodes))
        
        el_keys = ["eeg_fz", "eeg_cz", "eeg_pz"]
        for e_idx, el_key in enumerate(el_keys):
            r_el = r_electrodes[el_key]
            for n_idx in range(self.n_nodes):
                r_n = node_positions[n_idx]
                d_n = dipole_orientations[n_idx]
                
                # Cortical distance vector
                diff_c = r_el - r_n
                norm_c = np.linalg.norm(diff_c) + 1e-5
                # Thalamic vertical source deep below this column (R=0.15)
                r_t = 0.15 * r_n
                d_t = np.array([0.0, 0.0, 1.0])  # vertical deep dipole
                diff_t = r_el - r_t
                norm_t = np.linalg.norm(diff_t) + 1e-5
                
                lead_field_c[e_idx, n_idx] = np.dot(d_n, diff_c) / (norm_c ** 3)
                lead_field_t[e_idx, n_idx] = np.dot(d_t, diff_t) / (norm_t ** 3)

        # Normalize lead fields relative to Cz electrode for stability
        cz_c_max = np.max(lead_field_c[1, :]) or 1.0
        cz_t_max = np.max(lead_field_t[1, :]) or 1.0
        lead_field_c /= cz_c_max
        lead_field_t /= cz_t_max

        # Generate multi-channel signals by matrix projection
        ds_cortical = downsampled["cortical_pyramidal"] # shape (ds_steps, n_nodes)
        ds_spindle = downsampled["spindle"]            # shape (ds_steps, n_nodes)
        
        # Electrode potentials: sum of projections across all lattice nodes
        for e_idx, el_key in enumerate(el_keys):
            w_c = lead_field_c[e_idx, :]
            w_t = lead_field_t[e_idx, :]
            
            proj_c = np.dot(ds_cortical, w_c)
            proj_t = np.dot(ds_spindle, w_t)
            
            # Mix cortical and spindle components using default parameter weights
            downsampled[el_key] = proj_c + p.eeg_spindle_weight * proj_t

        return downsampled
