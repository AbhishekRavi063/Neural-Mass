import numpy as np
import time
import math
from numba import njit

@njit
def simulate_numba(steps, dt, p_array, noise_all):
    # p_array contains the parameters in a specific order:
    # 0: cortical_damping
    # 1: relay_to_reticular
    # 2: background_drive
    # 3: thalamus_to_cortex
    # 4: spindle_damping
    # 5: spindle_drive_offset
    # 6: neuromodulation_strength
    # 7: reticular_down_state_mix
    # 8: cortical_frequency
    # 9: spindle_frequency
    # 10: adaptation_strength
    # 11: cortical_inhibitory_weight
    # 12: thalamic_relay_damping
    # 13: cortical_excitation_scale
    # 14: cortex_to_thalamus
    # 15: reticular_inhibition_scale
    # 16: reticular_inhibition
    # 17: spindle_feedback_gain
    # 18: adaptation_tau
    # 19: spindle_reticular_mix
    # 20: noise_std
    
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
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_spindle_drive_offset
        
        k2_0 = u1
        k2_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2
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
        sp_drive = ds_gate + spindle_reticular_mix * ret_act - eff_spindle_drive_offset
        
        k3_0 = u1
        k3_1 = c1 * u1 + c2 * u0 - adaptation_strength * u4 - cortical_inhibitory_weight * i_cort + eff_thalamus_to_cortex * u2
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

p_array = np.array([
    0.18, # cortical_damping
    0.70, # relay_to_reticular
    0.20, # background_drive
    0.28, # thalamus_to_cortex
    0.55, # spindle_damping
    0.45, # spindle_drive_offset
    0.35, # neuromodulation_strength
    0.4,  # reticular_down_state_mix
    0.8,  # cortical_frequency
    13.0, # spindle_frequency
    0.45, # adaptation_strength
    0.35, # cortical_inhibitory_weight
    0.35, # thalamic_relay_damping
    18.0, # cortical_excitation_scale
    0.35, # cortex_to_thalamus
    14.0, # reticular_inhibition_scale
    0.55, # reticular_inhibition
    8.0,  # spindle_feedback_gain
    1.8,  # adaptation_tau
    0.25, # spindle_reticular_mix
    0.015 # noise_std
])

steps = 1000000 # 1000 seconds at dt=0.001 (1 million steps)
rng = np.random.default_rng(42)
noise_all = rng.normal(0.0, 1.0, size=(steps, 4))

print("Compiling Numba function...")
# Warm-up compile
simulate_numba(10, 0.001, p_array, noise_all[:10])

print("Benchmarking 1,000,000 steps...")
t0 = time.time()
res = simulate_numba(steps, 0.001, p_array, noise_all)
t1 = time.time()
print(f"Numba Implementation: {t1 - t0:.4f} seconds ({steps/(t1-t0):.1f} steps/sec)")
