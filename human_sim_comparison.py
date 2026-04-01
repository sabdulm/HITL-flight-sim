import numpy as np
import pandas as pd
import os
import glob
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION & IMPORTS ---
# Import your local metrics file
try:
    import metrics 
except ImportError:
    print("CRITICAL WARNING: 'metrics.py' not found. Ensure it is in the same directory.")

# Metadata paths
METADATA_FILE = "results/df_labeled_with_skill_score.csv" # Adjust extension if it's .pkl or .csv
DATA_ROOT = "flight_data"

# Metric keys we want to compare (Must match keys returned by your metrics.py)
# Adjust these based on exactly what your metrics.py returns
METRICS_TO_COMPARE = [
    "Inversion", 
    # "Flight_Path_Length_Success",
    'Waypoints',
    "Crashed",
    "CTE_Std", 
    "Control_Entropy", 
    "Action_Vol_Bang", 
    # "Time_Success", 
    "Energy_Variance",
    "PIO_Count_Pitch",
    "PIO_Count_Roll",
    "Max_G_Force"
]

# --- 2. DATA LOADING FUNCTIONS ---

def extract_episodes_from_npz(file_path, source_label, agent_id=None):
    """
    Parses a single .npz file containing multiple episodes (ep_{i}_obs, etc.)
    Returns a list of dictionaries (one per episode) with calculated metrics.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

    # Identify all unique episode indices 'i' by looking for 'ep_{i}_obs' keys
    # Filter keys that start with 'ep_' and end with '_obs'
    ep_keys = [k for k in data.files if k.startswith('ep_') and k.endswith('_obs')]
    
    episode_metrics_list = []

    for key in ep_keys:
        # Extract index 'i' from 'ep_i_obs'
        ep_idx = key.split('_')[1] 
        
        # Extract raw arrays
        obs = data[f"ep_{ep_idx}_obs"]
        rew = data[f"ep_{ep_idx}_rew"]
        act = data[f"ep_{ep_idx}_act"]
        info = data[f"ep_{ep_idx}_info"]
        # --- PARSE STATE ---
        ang_vel = obs[:, 0:3]     # P, Q, R
        quat = obs[:, 3:7]        # X, Y, Z, W (Fixed Mapping)
        lin_vel = obs[:, 7:10]    # u, v, w
        pos = obs[:, 10:13]       # x, y, z
        
        # --- BASIC METRICS ---
        if f"ep_{ep_idx}_global_targets" in data:
            targets = data[f"ep_{ep_idx}_global_targets"]
        else: targets = np.array([])
        # --- CRITICAL: CALCULATE METRICS HERE ---
        # We assume metrics.calculate_metrics(obs, act) returns a dict 
        # like {'CTE_Avg': 0.4, 'Control_Entropy': 1.2}
        duration = data[f"ep_{ep_idx}_real_duration"] if f"ep_{ep_idx}_real_duration" in data else len(rew)/60
        try:
            # If your metrics.py needs specific args, adjust here.
            # Passing raw arrays allows your functions to slice using OBS_LABELS logic.
            completed = metrics.calculate_success(info)
            calculated_metrics = {
                "Waypoints": metrics.calculate_waypoints_captured(rew),
                "Crashed": metrics.calculate_crashes(info, rew),
                "CTE_Std": metrics.calculate_cte_stats(pos, targets, rew)[0],
                "Control_Entropy": metrics.calculate_sample_entropy_fast(act),
                "Control_Entropy": metrics.calculate_control_entropy(act),
                "Action_Vol_Bang": metrics.calculate_volatility(act),
                "Time_Success": duration if completed else np.nan,
                "Energy_Variance": metrics.calculate_energy_variance(pos,lin_vel),
                # "Flight_Path_Length": metrics.calculate_flight_distance(obs),
                "Flight_Path_Length_Success": metrics.calculate_flight_distance(pos) if completed else np.nan,
                "Inversion": metrics.calculate_inverted_time(quat),
                "PIO_Count_Pitch": metrics.calculate_pio(act[:, 1]), # Assuming pitch
                "PIO_Count_Roll": metrics.calculate_pio(act[:, 0]),  # Assuming roll
                "Max_G_Force": metrics.calculate_max_g_force(lin_vel, ang_vel, quat)
            }
            
            
            # metrics.calculate_metrics(obs, act) 
        except AttributeError:
            # Fallback if specific function name is different
            print("Error: Define the correct function call to your metrics.py inside extract_episodes_from_npz")
            return []

        # Add Metadata
        calculated_metrics['Source'] = source_label # 'Human' or 'Agent'
        calculated_metrics['ID'] = agent_id if agent_id else 'Human_Expert'
        
        episode_metrics_list.append(calculated_metrics)
        
    return episode_metrics_list

def load_human_baseline(metadata_path, data_root):
    """
    Loads Expert + Alone human data using the directory structure:
    flight_data/{subject_id}/session1/task1/*.npz
    """
    print("Loading Human Data...")
    
    # Load Metadata (Assuming CSV, change to read_pickle if .pkl)
    if metadata_path.endswith('.csv'):
        df_meta = pd.read_csv(metadata_path)
    else:
        df_meta = pd.read_pickle(metadata_path)

    # Filter for Expert & Alone
    experts = df_meta[
        (df_meta['Skill_Label'].isin (['Expert'])) & 
        (df_meta['Condition'] == 'Alone-S1')
    ]['Subject'].unique()

    all_human_episodes = []

    for subj_id in experts:

        # Construct path: flight_data/{subject_id}/session1/task1/*.npz
        search_path = os.path.join(data_root, str(subj_id), "session1", "task1", "*.npz")
        files = glob.glob(search_path)
        
        for f in files:
            episodes = extract_episodes_from_npz(f, source_label="Human")
            all_human_episodes.extend(episodes)

    return pd.DataFrame(all_human_episodes)

def load_agent_data(agent_dict):
    """
    Loads agent data from the dictionary provided.
    agent_dict = { "Agent_Name": "path/to/file.npz" }
    """
    print("Loading Agent Data...")
    all_agent_episodes = []

    for agent_name, file_path in agent_dict.items():
        episodes = extract_episodes_from_npz(file_path, source_label="Agent", agent_id=agent_name)
        all_agent_episodes.extend(episodes)
        
    return pd.DataFrame(all_agent_episodes)

# --- 3. STATISTICAL COMPARISON (Wasserstein Distance) ---

def calculate_human_likeness(human_df, agent_df, metric_list):
    """
    Compares Agent distributions to Human distributions using Wasserstein Distance.
    Handles NaNs (e.g. crashed episodes) by dropping them per-metric.
    """
    results = []
    unique_agents = agent_df['ID'].unique()

    for agent in unique_agents:
        agent_scores = {}
        total_dist = 0
        valid_metric_count = 0
        
        for metric in metric_list:
            # --- 1. PREPARATION ---
            # Get valid Human data for this metric (remove NaNs)
            h_data = human_df[metric].dropna().values.reshape(-1, 1)
            # Get valid Agent data for this metric
            a_data = agent_df[agent_df['ID'] == agent][metric].dropna().values.reshape(-1, 1)
            # --- 2. SAFETY CHECKS ---
            # If Agent has NO valid data (e.g. 0% success rate -> no Time_Success),
            # we assign a "Max Penalty" because it is completely unlike the human.
            if len(a_data) == 0:
                penalty = 10.0 # Arbitrary high distance
                agent_scores[f"{metric}_Dist"] = penalty
                total_dist += penalty
                continue
                
            # If Humans have no data (shouldn't happen), skip
            if len(h_data) == 0:
                continue

            # --- 3. SCALING ---
            # Fit scaler ONLY on valid human data for this specific metric
            scaler = StandardScaler()
            scaler.fit(h_data)
            
            # Transform both
            h_scaled = scaler.transform(h_data).flatten()
            a_scaled = scaler.transform(a_data).flatten()
            
            # --- 4. CALCULATION ---
            dist = wasserstein_distance(h_scaled, a_scaled)
            
            agent_scores[f"{metric}_Dist"] = dist
            total_dist += dist
            valid_metric_count += 1
            
        summary = {
            'Agent_ID': agent,
            'Human_Likeness_Score': round(total_dist, 4), # Lower is better
            'Valid_Metrics': valid_metric_count
        }
        summary.update(agent_scores)
        results.append(summary)
    # Rank
    ranking_df = pd.DataFrame(results).sort_values('Human_Likeness_Score')
    ranking_df['Rank'] = range(1, len(ranking_df) + 1)
    return ranking_df



# --- 4. EXECUTION ---

if __name__ == "__main__":
    
    # 1. Define Agents (UPDATE THIS WITH YOUR ACTUAL PATHS)
    agent_dict = {
        "fw-ppo-v4": "agent_flight_data/log_20260216_185755.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_ALL_ALLData": "agent_flight_data/log_20260216_190257.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_ALL_SuccessOnly": "agent_flight_data/log_20260216_190759.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_Alone_ALLData": "agent_flight_data/log_20260216_191302.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_Alone_SuccessOnly": "agent_flight_data/log_20260216_191804.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_AloneArrow_ALLData": "agent_flight_data/log_20260216_192307.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_AloneArrow_SuccessOnly": "agent_flight_data/log_20260216_192810.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_AloneGhost_ALLData": "agent_flight_data/log_20260216_193312.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_AloneGhost_SuccessOnly": "agent_flight_data/log_20260216_193815.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_Arrow_ALLData": "agent_flight_data/log_20260216_194317.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_Arrow_SuccessOnly": "agent_flight_data/log_20260216_194820.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_ArrowGhost_ALLData": "agent_flight_data/log_20260216_195323.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_ArrowGhost_SuccessOnly": "agent_flight_data/log_20260216_195826.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_Ghost_ALLData": "agent_flight_data/log_20260216_200328.npz",
        "fw-ppo-v4-AIRL-v0/AIRL_Ghost_SuccessOnly": "agent_flight_data/log_20260216_200831.npz"
    }

    # 2. Load Data
    human_df = load_human_baseline(METADATA_FILE, DATA_ROOT)
    agent_df = load_agent_data(agent_dict)
    
    print(f"Loaded {len(human_df)} human episodes and {len(agent_df)} agent episodes.")

    # 3. Run Comparison
    # Ensure columns exist before running
    valid_metrics = [m for m in METRICS_TO_COMPARE if m in human_df.columns]
    
    ranking = calculate_human_likeness(human_df, agent_df, valid_metrics)
    
    print("\n--- AGENT HUMAN-LIKENESS RANKING (Lower Score = Better) ---")
    print(ranking[['Rank', 'Agent_ID', 'Human_Likeness_Score']])
    
    # Optional: Save detailed breakdown
    ranking.to_csv("agent_human_likeness_report.csv", index=False)