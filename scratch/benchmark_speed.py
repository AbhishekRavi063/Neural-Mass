import numpy as np
import time
import math

# Current numpy-based implementation
def derivatives_np(s, p, nm, external_stimulus):
    cortical_pyramidal, cortical_velocity, thalamic_relay, thalamic_velocity, adaptation, spindle_x, spindle_y = s
    
    spindle_bell = nm * (1.0 - nm)
    eff_cortical_damping = p['cortical_damping'] * max(0.35, 1.0 - 0.65 * nm)
    eff_relay_to_reticular = p['relay_to_reticular'] * (1.0 + p['neuromodulation_strength'] * nm)
    eff_background_drive = p['background_drive'] * max(0.15, 1.0 - 0.20 * nm)
    eff_thalamus_to_cortex = p['thalamus_to_cortex'] * (1.0 + 0.3 * p['neuromodulation_strength'] * nm)
    eff_spindle_damping = p['spindle_damping'] * max(0.30, 0.40 + 1.50 * spindle_bell)
    eff_spindle_drive_offset = p['spindle_drive_offset'] * (0.90 + 0.40 * (nm - 0.5) ** 2)

    # Sigmoid function inline
    cortical_excitation = 1.0 / (1.0 + np.exp(-3.0 * (cortical_pyramidal + eff_background_drive)))
    cortical_inhibition = 1.0 / (1.0 + np.exp(-4.0 * (cortical_excitation - 0.55)))
    down_state_gate = 1.0 - 1.0 / (1.0 + np.exp(-5.0 * (cortical_pyramidal - 0.05)))
    reticular_input = eff_relay_to_reticular * thalamic_relay + p['reticular_down_state_mix'] * down_state_gate
    reticular_activity = 1.0 / (1.0 + np.exp(-3.0 * reticular_input))

    slow_omega = 2.0 * np.pi * p['cortical_frequency']
    spindle_omega = 2.0 * np.pi * p['spindle_frequency']
    spindle_radius = spindle_x ** 2 + spindle_y ** 2
    spindle_drive = down_state_gate + p['spindle_reticular_mix'] * reticular_activity - eff_spindle_drive_offset

    dy = np.zeros(7)
    dy[0] = cortical_velocity
    dy[1] = (-2.0 * eff_cortical_damping * slow_omega * cortical_velocity - 
             (slow_omega ** 2) * cortical_pyramidal - 
             p['adaptation_strength'] * adaptation - 
             p['cortical_inhibitory_weight'] * cortical_inhibition + 
             eff_thalamus_to_cortex * thalamic_relay + 
             external_stimulus)
    dy[2] = thalamic_velocity
    dy[3] = (-2.0 * p['thalamic_relay_damping'] * spindle_omega * thalamic_velocity - 
             (spindle_omega ** 2) * thalamic_relay + 
             p['cortical_excitation_scale'] * p['cortex_to_thalamus'] * cortical_excitation - 
             p['reticular_inhibition_scale'] * p['reticular_inhibition'] * reticular_activity + 
             p['spindle_feedback_gain'] * spindle_x)
    dy[4] = (-adaptation + cortical_excitation) / p['adaptation_tau']
    dy[5] = eff_spindle_damping * spindle_drive * spindle_x - spindle_omega * spindle_y - spindle_radius * spindle_x
    dy[6] = spindle_omega * spindle_x + eff_spindle_damping * spindle_drive * spindle_y - spindle_radius * spindle_y
    return dy

def simulate_np(steps, dt, p, noise_all):
    state = np.zeros(7)
    state[0] = -0.25
    state[2] = 0.05
    state[5] = 0.01
    
    history = np.zeros((steps, 7))
    sqrt_dt = np.sqrt(dt)
    noise_std = p['noise_std']
    
    for idx in range(steps):
        s0 = state
        k1 = derivatives_np(s0, p, 0.6, 0.0)
        k2 = derivatives_np(s0 + 0.5 * dt * k1, p, 0.6, 0.0)
        k3 = derivatives_np(s0 + 0.5 * dt * k2, p, 0.6, 0.0)
        k4 = derivatives_np(s0 + dt * k3, p, 0.6, 0.0)
        
        raw_noise = noise_all[idx]
        noise_vec = np.zeros(7)
        noise_vec[1] = raw_noise[0] * noise_std
        noise_vec[3] = raw_noise[1] * noise_std
        noise_vec[5] = raw_noise[2] * noise_std * 0.1
        noise_vec[6] = raw_noise[3] * noise_std * 0.1
        
        state = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + noise_vec * sqrt_dt
        history[idx] = state
    return history

def simulate_scalar(steps, dt, p, noise_all):
    cortical_damping = p['cortical_damping']
    relay_to_reticular = p['relay_to_reticular']
    background_drive = p['background_drive']
    thalamus_to_cortex = p['thalamus_to_cortex']
    spindle_damping = p['spindle_damping']
    spindle_drive_offset = p['spindle_drive_offset']
    neuromodulation_strength = p['neuromodulation_strength']
    reticular_down_state_mix = p['reticular_down_state_mix']
    cortical_frequency = p['cortical_frequency']
    spindle_frequency = p['spindle_frequency']
    adaptation_strength = p['adaptation_strength']
    cortical_inhibitory_weight = p['cortical_inhibitory_weight']
    thalamic_relay_damping = p['thalamic_relay_damping']
    cortical_excitation_scale = p['cortical_excitation_scale']
    cortex_to_thalamus = p['cortex_to_thalamus']
    reticular_inhibition_scale = p['reticular_inhibition_scale']
    reticular_inhibition = p['reticular_inhibition']
    spindle_feedback_gain = p['spindle_feedback_gain']
    adaptation_tau = p['adaptation_tau']
    spindle_reticular_mix = p['spindle_reticular_mix']
    noise_std = p['noise_std']

    s0 = -0.25
    s1 = 0.0
    s2 = 0.05
    s3 = 0.0
    s4 = 0.0
    s5 = 0.01
    s6 = 0.0

    history = np.zeros((steps, 7))
    sqrt_dt = math.sqrt(dt)
    half_dt = 0.5 * dt
    dt_sixth = dt / 6.0
    
    nm = 0.6
    spindle_bell = nm * (1.0 - nm)
    eff_cortical_damping = cortical_damping * max(0.35, 1.0 - 0.65 * nm)
    eff_relay_to_reticular = relay_to_reticular * (1.0 + neuromodulation_strength * nm)
    eff_background_drive = background_drive * max(0.15, 1.0 - 0.20 * nm)
    eff_thalamus_to_cortex = thalamus_to_cortex * (1.0 + 0.3 * neuromodulation_strength * nm)
    eff_spindle_damping = spindle_damping * max(0.30, 0.40 + 1.50 * spindle_bell)
    eff_spindle_drive_offset = spindle_drive_offset * (0.90 + 0.40 * (nm - 0.5) ** 2)

    slow_omega = 2.0 * math.pi * cortical_frequency
    spindle_omega = 2.0 * math.pi * spindle_frequency
    
    c1 = -2.0 * eff_cortical_damping * slow_omega
    c2 = -(slow_omega ** 2)
    c3 = -2.0 * thalamic_relay_damping * spindle_omega
    c4 = -(spindle_omega ** 2)
    c5 = cortical_excitation_scale * cortex_to_thalamus
    c6 = -reticular_inhibition_scale * reticular_inhibition

    for idx in range(steps):
        # Step 1: k1
        # derivatives(s0, s1, s2, s3, s4, s5, s6)
        e_cort = 1.0 / (1.0 + math.exp(-3.0 * (s0 + eff_background_drive)))
        i_cort = 1.0 / (1.0 + math.exp(-4.0 * (e_cort - 0.55)))
        ds_gate = 1.0 - 1.0 / (1.0 + math.exp(-5.0 * (s0 - 0.05)))
        ret_in = eff_relay_to_reticular * s2 + reticular_down_state_mix * ds_gate
        ret_act = 1.0 / (1.0 + math.exp(-3.0 * ret_in))
        
        sp_rad = s5 * s5 + s6 * s6
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_spindle_drive_offset
        
        k1_0 = s1
        k1_1 = c1 * s1 + c2 * s0 - adaptation_strength * s4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * s2
        k1_2 = s3
        k1_3 = c3 * s3 + c4 * s2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * s5
        k1_4 = (-s4 + e_cort) / adaptation_tau
        k1_5 = eff_spindle_damping * sp_drive * s5 - spindle_omega * s6 - sp_rad * s5
        k1_6 = spindle_omega * s5 + eff_spindle_damping * sp_drive * s6 - sp_rad * s6

        # Step 2: k2 at state + half_dt * k1
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
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_spindle_drive_offset
        
        k2_0 = u1
        k2_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2
        k2_2 = u3
        k2_3 = c3 * u3 + c4 * u2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * u5
        k2_4 = (-u4 + e_cort) / adaptation_tau
        k2_5 = eff_spindle_damping * sp_drive * u5 - spindle_omega * u6 - sp_rad * u5
        k2_6 = spindle_omega * u5 + eff_spindle_damping * sp_drive * u6 - sp_rad * u6

        # Step 3: k3 at state + half_dt * k2
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
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_spindle_drive_offset
        
        k3_0 = u1
        k3_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2
        k3_2 = u3
        k3_3 = c3 * u3 + c4 * u2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * u5
        k3_4 = (-u4 + e_cort) / adaptation_tau
        k3_5 = eff_spindle_damping * sp_drive * u5 - spindle_omega * u6 - sp_rad * u5
        k3_6 = spindle_omega * u5 + eff_spindle_damping * sp_drive * u6 - sp_rad * u6

        # Step 4: k4 at state + dt * k3
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
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_spindle_drive_offset
        
        k4_0 = u1
        k4_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2
        k4_2 = u3
        k4_3 = c3 * u3 + c4 * u2 + c5 * e_cort + c6 * ret_act + spindle_feedback_gain * u5
        k4_4 = (-u4 + e_cort) / adaptation_tau
        k4_5 = eff_spindle_damping * sp_drive * u5 - spindle_omega * u6 - sp_rad * u5
        k4_6 = spindle_omega * u5 + eff_spindle_damping * sp_drive * u6 - sp_rad * u6

        # Noise step
        raw_noise = noise_all[idx]
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

# Setup parameters
p = {
    'cortical_frequency': 0.8,
    'spindle_frequency': 13.0,
    'cortical_damping': 0.18,
    'spindle_damping': 0.55,
    'adaptation_strength': 0.45,
    'adaptation_tau': 1.8,
    'cortex_to_thalamus': 0.35,
    'thalamus_to_cortex': 0.28,
    'reticular_inhibition': 0.55,
    'relay_to_reticular': 0.70,
    'background_drive': 0.20,
    'noise_std': 0.015,
    'cortical_excitation_scale': 18.0,
    'reticular_inhibition_scale': 14.0,
    'spindle_feedback_gain': 8.0,
    'cortical_inhibitory_weight': 0.35,
    'thalamic_relay_damping': 0.35,
    'reticular_down_state_mix': 0.4,
    'spindle_reticular_mix': 0.25,
    'spindle_drive_offset': 0.45,
    'eeg_spindle_weight': 0.18,
    'eeg_relay_weight': 0.08,
    'neuromodulator_level': 0.6,
    'neuromodulation_strength': 0.35
}

steps = 100000 # 100 seconds at dt=0.001
rng = np.random.default_rng(42)
noise_all = rng.normal(0.0, 1.0, size=(steps, 4))

print("Benchmarking...")
t0 = time.time()
res1 = simulate_np(steps, 0.001, p, noise_all)
t1 = time.time()
print(f"NumPy Implementation:  {t1 - t0:.4f} seconds")

t2 = time.time()
res2 = simulate_scalar(steps, 0.001, p, noise_all)
t3 = time.time()
print(f"Scalar Implementation: {t3 - t2:.4f} seconds")

# Assert results are close
np.testing.assert_allclose(res1, res2, rtol=1e-5, atol=1e-5)
print("Results match! Speedup: {:.2f}x".format((t1 - t0) / (t3 - t2)))
