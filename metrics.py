import numpy as np
import pandas as pd
import antropy as ant


# The total number of target waypoints successfully triggered by the aircraft during the episode.
def calculate_waypoints_captured(rewards, threshold=90.0):
    return np.sum(rewards >= threshold)

# Binary flag indicating if the pilot completed the entire course (1 = Success, 0 = Fail).
def calculate_success(infos):
    if len(infos) == 0: return 0
    # Checks the last frame's info dict for the completion flag
    # Assumes PyFlyt standard key 'env_complete'
    return 1 if infos[-1].get('env_complete', False) else 0


# Binary flag indicating if the flight ended in a collision.
def calculate_crashes(infos, rews, threshold=-90.0):
    if len(infos) == 0: return 0
    # Checks the last frame for collision flag
    # Assumes PyFlyt standard key 'collision'
    return 1 if infos[-1].get('collision', False) or np.sum(rews <= threshold) > 0 else 0

# The count of times the aircraft entered the "Warning Zone" of the active waypoint 
# but exited without capturing it (failed attempts).
def calculate_near_misses(positions, waypoints, rewards, warn_dist=10.0):
    if len(waypoints) == 0: return 0
    
    near_miss_count = 0
    active_idx = 0
    in_zone = False # State flag: Are we currently inside the bubble?
    
    for t, pos in enumerate(positions):
        # Stop if we have finished all waypoints
        if active_idx >= len(waypoints): break
        
        # 1. Check for Capture Event (Reward spike)
        if rewards[t] >= 90.0:
            # We captured it! 
            # Even if we were 'in_zone', this is a success, not a miss.
            active_idx += 1
            in_zone = False # Reset state for the NEXT waypoint
            continue
            
        # 2. Check Distance to the CURRENT Active Waypoint
        target = waypoints[active_idx]
        dist = np.linalg.norm(pos - target)
        
        if dist < warn_dist:
            # We are inside the warning bubble
            if not in_zone:
                in_zone = True # Mark entry
        else:
            # We are outside the warning bubble
            if in_zone:
                # We WERE inside, but now we left, and we did NOT capture it.
                # This counts as a "Missed Pass" or "Failed Approach"
                near_miss_count += 1
                in_zone = False # Reset state
                
    return near_miss_count


def calculate_near_crashes(positions, floor_threshold=5.0, crash_tolerance=0.1):
    """
    Counts the number of times the aircraft entered a dangerous low altitude 
    and recovered without crashing.
    
    Args:
        positions (np.array): Array of shape (N, 3) containing [x, y, z].
        floor_threshold (float): Altitude below which is considered "Dangerous".
        crash_tolerance (float): Altitude below which is considered a "Crash".
        
    Returns:
        int: Count of successful recoveries from low altitude.
    """
    altitudes = positions[:, 2] # Assuming Z-axis is altitude
    near_crash_count = 0
    in_danger_zone = False # State flag: Are we currently dangerously low?
    
    for alt in altitudes:
        # 1. Check for Crash
        if alt <= crash_tolerance:
            # We hit the ground. This cancels the current "Danger" state.
            # A crash is not a "near miss", it is a failure.
            in_danger_zone = False 
            continue

        # 2. Check Danger Zone Status
        if alt < floor_threshold:
            # We are dangerously low (but haven't crashed yet)
            if not in_danger_zone:
                in_danger_zone = True # Mark entry into danger zone
        else:
            # We are safe (above threshold)
            if in_danger_zone:
                # We WERE in danger, and now we are safe again.
                # We did NOT crash in between. This counts as a recovery.
                near_crash_count += 1
                in_danger_zone = False # Reset state
                
    return near_crash_count


# The total duration of the episode, valid only if the task was successfully completed.
def calculate_time_to_completion(duration, is_success):
    if not is_success:
        return np.nan  # Or None
    return duration

# The average perpendicular distance between the aircraft and the ideal straight-line path between the previous and active waypoint.
def calculate_cte(positions, waypoints, rewards):
    if len(waypoints) == 0: return 0.0
    cte_sum, count = 0.0, 0
    prev_wp = np.array([0.0, 0.0, 10.0]) # Start
    active_idx = 0

    for t, pos in enumerate(positions):
        if active_idx >= len(waypoints): break
        if rewards[t] >= 90.0:
            prev_wp = waypoints[active_idx]
            active_idx += 1
            continue

        path_vec = waypoints[active_idx] - prev_wp
        drone_vec = pos - prev_wp
        path_len = np.linalg.norm(path_vec)

        if path_len < 1e-3: dist = np.linalg.norm(pos - waypoints[active_idx])
        else: dist = np.linalg.norm(np.cross(path_vec, drone_vec)) / path_len

        cte_sum += dist
        count += 1
    return cte_sum / count if count > 0 else 0.0


def calculate_cte_stats(positions, waypoints, rewards):
    """
    Calculates Cross-Track Error (CTE) statistics using segment clamping.
    This prevents the "overshoot" error by measuring distance to the segment, 
    not the infinite line.
    """
    if len(waypoints) == 0: return 0.0, 0.0
    
    cte_values = []
    prev_wp = np.array([0.0, 0.0, 10.0]) # Starting position
    active_idx = 0

    for t, pos in enumerate(positions):
        if active_idx >= len(waypoints): break
        
        if rewards[t] >= 90.0:
            prev_wp = waypoints[active_idx]
            active_idx += 1
            continue

        # Segment AB
        A = prev_wp
        B = waypoints[active_idx]
        AB = B - A
        AP = pos - A
        
        # Project AP onto AB to find normalized distance 't'
        # Clamp t between 0 and 1 to stay on the segment
        denom = np.dot(AB, AB)
        if denom < 1e-6:
            dist = np.linalg.norm(pos - B)
        else:
            t_proj = np.clip(np.dot(AP, AB) / denom, 0, 1)
            closest_point = A + t_proj * AB
            dist = np.linalg.norm(pos - closest_point)

        cte_values.append(dist)

    if not cte_values: return 0.0, 0.0
    
    return np.std(cte_values), np.mean(cte_values)
# The total Euclidean distance flown by the aircraft.
def calculate_flight_distance(positions):
    if len(positions) < 2: return 0.0
    return np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))

# The percentage of flight time where the aircraft’s bank angle exceeded 90 degrees (inverted).
def calculate_inverted_time(quaternions):
    # Quat [x, y, z, w]. Up_z = 1 - 2(x^2 + y^2)
    qx, qy = quaternions[:, 0], quaternions[:, 1]
    up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    return (np.sum(up_z < 0.0) / len(quaternions)) * 100.0

def calculate_max_g_force(lin_vel, ang_vel, quaternions):
    """
    Calculates Vertical Load Factor (n_z) with Gravity Compensation.
    nz = (u*q - v*p)/g + cos(theta)cos(phi)
    """
    if len(lin_vel) == 0: return 1.0
    g = 9.81
    
    # Extract velocities and rates
    u = lin_vel[:, 0] # Forward speed
    v = lin_vel[:, 1] # Lateral speed
    p = ang_vel[:, 0] # Roll rate
    q = ang_vel[:, 1] # Pitch rate
    
    # Calculate orientation-based gravity component (Down-vector in Body Frame)
    # Using quaternions [x, y, z, w] to find the body-Z component of the gravity vector
    qx, qy, qz, qw = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # This represents the projection of the gravity vector onto the aircraft's vertical axis
    # In level flight, gravity_component = 1.0. Inverted, it's -1.0.
    gravity_component = 1.0 - 2.0 * (qx**2 + qy**2)
    
    # Vertical acceleration in body frame (centripetal + gravity)
    # n_z = (Vertical acceleration) / g
    # Simplified aerodynamic load factor + gravity component
    accel_z = (u * q - v * p) 
    vertical_gs = (accel_z / g) + gravity_component

    return np.max(np.abs(vertical_gs))

import antropy as ant

def calculate_sample_entropy_fast(actions):
    """
    Uses the AntroPy package for high-speed Sample Entropy calculation.
    """
    # Isolate the Roll or Pitch channel
    # Antropy expects a 1D array
    signal = np.linalg.norm(actions[:, :2], axis=1)
    
    # order corresponds to m (template length)
    # tolerance corresponds to r (0.2 * std is the default in antropy)
    try:
        se = ant.sample_entropy(signal, order=2)
    except Exception:
        # Handles cases where the signal has 0 variance or is too short
        se = 0.0
        
    return se
# The Shannon entropy of the pilot's joystick inputs, representing "control disorder."
from scipy.stats import entropy
def calculate_control_entropy(actions):
    # Magnitude of Roll/Pitch stick deflection
    magnitudes = np.linalg.norm(actions[:, :2], axis=1)
    hist, _ = np.histogram(magnitudes, bins=20, range=(0, 1), density=True)
    return entropy(hist + 1e-10, base=2)

# The count of rapid stick reversals (sign flips) in the control input derivative.
def calculate_pio(action_channel):
    deltas = action_channel[1:] - action_channel[:-1]
    signs = np.sign(deltas)
    signs[np.abs(deltas) < 0.01] = 0 # Filter noise
    signs_clean = signs[signs != 0]
    if len(signs_clean) < 2: return 0
    return np.sum(signs_clean[1:] * signs_clean[:-1] < 0)

# The percentage of time the control stick is saturated (>90% deflection).
def calculate_volatility(actions):
    is_bang = np.any(np.abs(actions[:, :2]) > 0.9, axis=1)
    return (np.sum(is_bang) / len(actions)) * 100.0

# The variance of the specific energy state, measuring energy management stability.
def calculate_energy_variance(positions, lin_vel):
    G = 9.81
    h = positions[:, 2]
    v = np.linalg.norm(lin_vel, axis=1)
    Es = h + (v**2) / (2 * G)
    return np.var(Es)

# The cosine similarity between the Human's control vector and the AI's control vector.
def calculate_trust_cosine(human_act, ai_act):
    h, a = human_act[:, :2], ai_act[:, :2]
    dot = np.sum(h * a, axis=1)
    norms = (np.linalg.norm(h, axis=1) * np.linalg.norm(a, axis=1)) + 1e-6
    return np.mean(dot / norms)

# The time lag that maximizes the cross-correlation between Human and AI signals.
from scipy.signal import correlate
def calculate_latency(h_sig, a_sig, fps=60.0):
    # Normalize
    h_norm = (h_sig - np.mean(h_sig)) / (np.std(h_sig) + 1e-6)
    a_norm = (a_sig - np.mean(a_sig)) / (np.std(a_sig) + 1e-6)
    corr = correlate(h_norm, a_norm, mode='full')
    lags = np.arange(-len(h_norm) + 1, len(h_norm))
    lag_frames = lags[np.argmax(corr)]
    return (lag_frames / fps) * 1000.0 # ms

# The difference in control magnitude between the Human and the AI. Positive values indicate the Human is pushing harder (aggressive); negative values indicate the Human is pushing softer (timid) than the AI.
import numpy as np

def calculate_trust_intensity_gap(human_act, ai_act, condition, infos):
    """
    Calculates Control Intensity Gap (Gi).
    Filters for active assist frames if condition is Ghost-S2-adaptive.
    """
    # 1. Handle Adaptive Masking
    if "adaptive" in condition:
        # Extract assist state from infos list of dicts
        assist_active = np.array([i.get("assist_state", 0) for i in infos])
        mask = (assist_active == 1)
        
        # If the assist never triggered, we can't measure interaction
        if not np.any(mask):
            return {k: np.nan for k in ["intensity_gap_roll", "intensity_gap_pitch", "intensity_gap_global"]}
        
        h_act = human_act[mask]
        a_act = ai_act[mask]
    else:
        h_act = human_act
        a_act = ai_act

    # 2. Extract channels
    h_roll, h_pitch = h_act[:, 0], h_act[:, 1]
    a_roll, a_pitch = a_act[:, 0], a_act[:, 1]
    
    # 3. Calculate Axis Gaps
    gap_roll = np.abs(h_roll) - np.abs(a_roll)
    gap_pitch = np.abs(h_pitch) - np.abs(a_pitch)
    
    # 4. Global Magnitude Gap (Euclidean)
    mag_h = np.linalg.norm(h_act[:, :2], axis=1)
    mag_a = np.linalg.norm(a_act[:, :2], axis=1)
    gap_global = mag_h - mag_a

    return {
        "intensity_gap_roll": np.mean(gap_roll),
        "intensity_gap_pitch": np.mean(gap_pitch),
        "intensity_gap_global": np.mean(gap_global)
    }

def calculate_trust_sign_similarity(human_act, ai_act, condition, infos, deadzone=0.05):
    """
    Calculates Sign Similarity percentage.
    Filters for active assist frames if condition is Ghost-S2-adaptive.
    """
    # 1. Handle Adaptive Masking
    if "adaptive" in condition:
        assist_active = np.array([i.get("assist_state", 0) for i in infos])
        mask = (assist_active == 1)
        
        if not np.any(mask):
            return {k: np.nan for k in ["sign_sim_roll", "sign_sim_pitch", "sign_sim_global"]}
            
        h_act = human_act[mask]
        a_act = ai_act[mask]
    else:
        h_act = human_act
        a_act = ai_act

    # 2. Extract Channels
    h_roll, h_pitch = h_act[:, 0], h_act[:, 1]
    a_roll, a_pitch = a_act[:, 0], a_act[:, 1]

    def get_clean_sign(signal):
        signs = np.sign(signal)
        signs[np.abs(signal) < deadzone] = 0
        return signs

    s_h_r, s_a_r = get_clean_sign(h_roll), get_clean_sign(a_roll)
    s_h_p, s_a_p = get_clean_sign(h_pitch), get_clean_sign(a_pitch)

    # 3. Matches
    match_roll = (s_h_r == s_a_r)
    match_pitch = (s_h_p == s_a_p)
    global_intent = match_roll & match_pitch

    return {
        "sign_sim_roll": np.mean(match_roll) * 100.0,
        "sign_sim_pitch": np.mean(match_pitch) * 100.0,
        "sign_sim_global": np.mean(global_intent) * 100.0
    }