# run_comparison.py

import torch
import numpy as np
import pickle
import pandas as pd
import os
import time
from environment.env import AirGroundEnv
from marl.mappo_agent import MAPPOAgent
from marl.mappo_agent_mlp import MAPPOAgentMLP
from marl.qmix_agent import QMIXAgent
from marl.ippo_agent import IPPOAgent
from marl.maddpg_agent import MADDPGAgent
from baselines import RandomAgent, HeuristicAgentV4_Expert
from marl.mappo_agent_ablation1 import MAPPOAgentAblation1
from marl.mappo_agent_ablation2 import MAPPOAgentAblation2
from marl.mappo_agent_ablation4 import MAPPOAgentAblation4

import config

POWER_LEVELS_dBm = np.linspace(0, config.NODE_TRANSMIT_POWER_dBm, config.ACTION_POWER_LEVELS)
UAV_MOVE_ACTIONS = {0: [0, 0, 0], 1: [0, 5, 0], 2: [0, -5, 0], 3: [5, 0, 0], 4: [-5, 0, 0], 5: [0, 0, 5], 6: [0, 0, -5]}
UGV_MOVE_ACTIONS = {0: [0, 0, 0], 1: [0, 3, 0], 2: [0, -3, 0], 3: [3, 0, 0], 4: [-3, 0, 0]}
UAV_ACTION_DIM = len(UAV_MOVE_ACTIONS) * config.ACTION_POWER_LEVELS
UGV_ACTION_DIM = len(UGV_MOVE_ACTIONS) * config.ACTION_POWER_LEVELS


def map_action_id_to_env_action(agent_id, action_id, num_uavs):
    if agent_id < num_uavs:
        power_idx = action_id % config.ACTION_POWER_LEVELS
        move_idx = action_id // config.ACTION_POWER_LEVELS
        accel = UAV_MOVE_ACTIONS[move_idx]
    else:
        power_idx = action_id % config.ACTION_POWER_LEVELS
        move_idx = action_id // config.ACTION_POWER_LEVELS
        accel = UGV_MOVE_ACTIONS[move_idx]
    transmit_power_watts = 10 ** ((POWER_LEVELS_dBm[power_idx] - 30) / 10)
    return {'acceleration': accel, 'transmit_power_watts': transmit_power_watts}


def process_obs_for_flat_agents(obs_list):
    processed = []
    max_n, feat_d = config.MAX_OBS_NODES, config.NODE_FEATURE_DIM
    for o in obs_list:
        feat = o.get('node_features', np.array([]))
        padded = np.zeros((max_n, feat_d), dtype=np.float32)
        if feat.size > 0:
            n, f = feat.shape
            copy_n, copy_f = min(n, max_n), min(f, feat_d)
            padded[:copy_n, :copy_f] = feat[:copy_n, :copy_f]
        processed.append(padded.flatten())
    return processed


def process_obs_for_ablation5(obs_list):
    processed_obs_list = []
    for agent_obs in obs_list:
        new_agent_obs = {
            'edge_index': agent_obs['edge_index'],
            'edge_attr': agent_obs['edge_attr'],
            'node_features': agent_obs['node_features'][:, :-2]
        }
        processed_obs_list.append(new_agent_obs)
    return processed_obs_list


def calculate_composite_scores(results):
    print("\n--- Calculating the composite weighted score ---")
    target_models = []
    raw_values = {'thr': [], 'conn': [], 'eng': []}

    for name, stats in results.items():
        if "Baseline" in name: continue
        target_models.append(name)
        raw_values['thr'].append(stats['throughput']['mean'])
        raw_values['conn'].append(stats['connectivity']['mean'])
        raw_values['eng'].append(stats['energy_cost']['mean'])

    if not target_models: return {}

    vals = {k: np.array(v) for k, v in raw_values.items()}
    norm_vals = {}
    for k in ['thr', 'conn', 'eng']:
        v_min, v_max = np.min(vals[k]), np.max(vals[k])
        if v_max - v_min < 1e-9:
            norm_vals[k] = np.zeros_like(vals[k])
        else:
            norm_vals[k] = (vals[k] - v_min) / (v_max - v_min)

    weights = config.CURRENT_REWARD_WEIGHTS
    w_sum = sum(weights.values())
    w_thr, w_conn, w_eng = weights['throughput'] / w_sum, weights['connectivity'] / w_sum, weights['energy'] / w_sum

    final_scores = {}
    for i, name in enumerate(target_models):
        score = w_thr * norm_vals['thr'][i] + w_conn * norm_vals['conn'][i] + w_eng * norm_vals['eng'][i]
        final_scores[name] = score
    return final_scores


def generate_comparison_excel(results, scores, filename):
    print(f"--- Generating data report: {filename} ---")

    row_labels = [
        "Throughput (Mean)", "Throughput (Std)",
        "Connectivity (Mean)", "Connectivity (Std)",
        "Energy Cost (Mean)", "Energy Cost (Std)",
        "Composite Score (Normalized)"
    ]

    data_dict = {"Metrics": row_labels}

    for agent_name, stats in results.items():
        score_val = scores.get(agent_name, 0.0)

        col_data = [
            round(stats['throughput']['mean'], 8),
            round(stats['throughput']['std'], 8),
            round(stats['connectivity']['mean'], 8),
            round(stats['connectivity']['std'], 8),
            round(stats['energy_cost']['mean'], 8),
            round(stats['energy_cost']['std'], 8),
            round(score_val, 8)
        ]
        data_dict[agent_name] = col_data

    df = pd.DataFrame(data_dict)

    try:
        df.to_excel(filename, index=False)
        print(f">>> Success! The Excel file has been generated: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"!!!Excel generation failed: {e}")
        print("Possible reasons: 1. The openpyxl library is not installed (pip install openpyxl) 2. The file is currently in use by another program")

        csv_filename = filename.replace('.xlsx', '.csv')
        try:
            df.to_csv(csv_filename, index=False)
            print(f">>> Backup plan enabled: Data saved as CSV file: {csv_filename}")
        except Exception as e2:
            print(f"!!!CSV backup also failed: {e2}")


def print_text_report(results, scores):
    print("\n" + "=" * 100 + "\n" + " " * 40 + "Comparison Results (Mean ± Std)\n" + "=" * 100)
    print(
        f"{'Model':<25} | {'Throughput':>22} | {'Connectivity':>22} | {'Energy Cost':>22} | {'Score':>10}\n" + "-" * 108)
    for name, s in results.items():
        score = scores.get(name, "N/A");
        score_str = f"{score:.8f}" if isinstance(score, float) else f"{score:>10}"
        print(f"{name:<25} | {s['throughput']['mean']:10.8f} ± {s['throughput']['std']:.8f} | "
              f"{s['connectivity']['mean']:10.8f} ± {s['connectivity']['std']:.8f} | "
              f"{s['energy_cost']['mean']:10.8f} ± {s['energy_cost']['std']:.8f} | {score_str}")
    print("=" * 108)


def run_evaluation(agent, env, num_episodes, agent_name, agent_type, scenarios):
    all_reward_components = []
    print(f"\n--- Begin evaluating the model: {agent_name} ---")

    is_flat = agent_type in ["MLP", "QMIX"]
    is_ablation5 = agent_type == "ABLATION5"

    start_time = time.time()
    for ep in range(num_episodes):
        scenario = scenarios[ep] if scenarios else None
        obs, _ = env.reset(scenario_config=scenario)

        if hasattr(agent, 'num_agents') and agent.num_agents != env.num_agents:
            agent.num_agents = env.num_agents
        elif hasattr(agent, 'n_agents') and agent.n_agents != env.num_agents:
            agent.n_agents = env.num_agents

        if is_flat: flat_obs = process_obs_for_flat_agents(obs)
        if is_ablation5: obs = process_obs_for_ablation5(obs)

        hidden_states = [torch.zeros(1, config.RNN_HIDDEN_DIM) for _ in range(env.num_agents)]
        ep_stats = {'throughput': 0, 'connectivity': 0, 'energy_cost': 0}

        for step in range(config.EPISODE_LENGTH):
            avail = env.get_avail_actions()

            if agent_type in ["STGAT", "IPPO", "ABLATION5", "ABLATION2"]:
                actions, _, new_hs = agent.choose_action(obs, hidden_states, avail, deterministic=True)
                hidden_states = new_hs
            elif agent_type in ["ABLATION1", "ABLATION4"]:
                actions, _, _ = agent.choose_action(obs, None, avail, deterministic=True)
            elif agent_type == "MLP":
                actions, new_hs = [], []
                for i in range(env.num_agents):
                    a, _, h = agent.choose_action(flat_obs[i], hidden_states[i], avail[i], deterministic=True)
                    actions.append(a);
                    new_hs.append(h)
                hidden_states = new_hs
            elif agent_type == "QMIX":
                actions, raw_h = agent.choose_action(flat_obs, hidden_states, avail)
                hidden_states = [h.unsqueeze(0) for h in raw_h]
            elif agent_type == "MADDPG":
                actions, hidden_states = agent.choose_action(obs, hidden_states, avail, explore=False)
            elif isinstance(agent, RandomAgent):
                actions = agent.choose_action(avail)
            elif isinstance(agent, HeuristicAgentV4_Expert):
                agent.nodes = env.nodes;
                actions = agent.choose_action(obs)

            env_actions = {n.id: map_action_id_to_env_action(i, a, env.num_uavs) for i, (n, a) in enumerate(zip(env.nodes, actions))}
            obs, _, _, done, info = env.step(env_actions)

            if is_flat: flat_obs = process_obs_for_flat_agents(obs)
            if is_ablation5: obs = process_obs_for_ablation5(obs)

            for k in ep_stats: ep_stats[k] += info.get(k, 0)
            if done: break

        all_reward_components.append(ep_stats)
        if (ep + 1) % 10 == 0: print(f"  Progress: {ep + 1}/{num_episodes} | Elapsed time: {time.time() - start_time:.1f}s")

    results = {}
    for k in ['throughput', 'connectivity', 'energy_cost']:
        vals = [ep[k] for ep in all_reward_components]
        results[k] = {'mean': np.mean(vals), 'std': np.std(vals)}
    return results


def load_stgat_agent(ep):
    a = MAPPOAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    path = 'models_env_2_2_6'
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_mlp_agent(ep):
    obs_dim = config.MAX_OBS_NODES * config.NODE_FEATURE_DIM
    a = MAPPOAgentMLP(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM, obs_dim)
    path = 'models_mlp_env_2_2_6'
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_maddpg_agent(ep):
    a = MADDPGAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM, config.NUM_UAVS)
    try:
        a.load_models('models_maddpg_env_2_2_6', ep); return a
    except:
        return None


def load_qmix_agent(ep):
    obs_dim = config.MAX_OBS_NODES * config.NODE_FEATURE_DIM
    state_dim = (config.NUM_UAVS + config.NUM_UGVS) * obs_dim
    a = QMIXAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, state_dim, obs_dim)
    try:
        a.load('models_qmix_env_2_2_6', ep); return a
    except:
        return None


def load_ippo_agent(ep):
    a = IPPOAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    path = 'models_ippo_env_2_2_6'
    try:
        a.load_models(f'{path}/actor_ippo_{ep}.pth', f'{path}/critic_ippo_{ep}.pth',
                      f'{path}/encoder_ippo_{ep}.pth'); return a
    except:
        return None


def load_ablation1_agent(ep):  # S-GAT-MAPPO (No RNN)
    a = MAPPOAgentAblation1(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    path = 'models_ablation1_env_2_2_6'
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_ablation2_agent(ep):  # ST-GCN-MAPPO (GAT->GCN)
    a = MAPPOAgentAblation2(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    path = 'models_ablation2_env_2_2_6'
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_ablation4_agent(ep):  # S-GCN-MAPPO (No RNN + GCN)
    a = MAPPOAgentAblation4(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    path = 'models_ablation4_env_2_2_6'
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_ablation5_agent(ep):  # STGAT-MAPPO (No Hetero)
    orig_dim = config.NODE_FEATURE_DIM
    config.NODE_FEATURE_DIM = orig_dim - 2
    a = MAPPOAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    config.NODE_FEATURE_DIM = orig_dim  # 恢复

    path = 'models_ablation5_env_2_2_6'
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_heuristic(env): return HeuristicAgentV4_Expert(env.num_agents, env.nodes, config.ACTION_POWER_LEVELS,
                                                        UAV_MOVE_ACTIONS, env.num_uavs, env.num_ugvs, env.area_size)


def load_random(env): return RandomAgent(env.num_agents, UAV_ACTION_DIM, UGV_ACTION_DIM, env.num_uavs)


def main():
    DATA_MODE = 'RUN_NEW'

    # Comparison Mode: 'BASELINE' or 'ABLATION'
    COMPARISON_MODE = 'ABLATION'

    if COMPARISON_MODE == 'BASELINE':
        PKL_FILE = 'comparison_baseline.pkl'
        XLS_FILE = 'comparison_baseline.xlsx'
    else:
        PKL_FILE = 'comparison_ablation.pkl'
        XLS_FILE = 'comparison_ablation.xlsx'

    EVAL_EPISODES = 100
    SCENARIO_FILE = 'evaluation_scenarios.pkl'
    # ===========================================

    OUR_MODEL = {
        "STGAT-MAPPO (Ours)": {"run": True, "loader": load_stgat_agent, "ep": 6720, "type": "STGAT"}
    }

    BASELINE_MODELS = {
        "MAPPO-MLP": {"run": False, "loader": load_mlp_agent, "ep": 9600, "type": "MLP"},
        "QMIX-STGAT": {"run": False, "loader": load_qmix_agent, "ep": 1500, "type": "QMIX"},
        "IPPO-STGAT": {"run": False, "loader": load_ippo_agent, "ep": 6720, "type": "IPPO"},
        "MADDPG-STGAT": {"run": False, "loader": load_maddpg_agent, "ep": 45000, "type": "MADDPG"},
        "Heuristic": {"run": False, "loader": load_heuristic, "ep": None, "type": "HEURISTIC"},
        "Random": {"run": True, "loader": load_random, "ep": None, "type": "RANDOM"},
    }

    ABLATION_MODELS = {
        "S-GAT-MAPPO": {"run": True, "loader": load_ablation1_agent, "ep": 2400, "type": "ABLATION1"},
        "ST-GCN-MAPPO": {"run": True, "loader": load_ablation2_agent, "ep": 1680, "type": "ABLATION2"},
        "S-GCN-MAPPO": {"run": True, "loader": load_ablation4_agent, "ep":  2640, "type": "ABLATION4"},
        "STGAT-MAPPO (No Hetero)": {"run": True, "loader": load_ablation5_agent, "ep": 3600, "type": "ABLATION5"},
    }

    AGENTS_TO_RUN = OUR_MODEL.copy()
    if COMPARISON_MODE == 'BASELINE':
        AGENTS_TO_RUN.update(BASELINE_MODELS)
    else:
        AGENTS_TO_RUN.update(ABLATION_MODELS)

    final_results = {}

    if DATA_MODE == 'LOAD_ONLY':
        if os.path.exists(PKL_FILE):
            with open(PKL_FILE, 'rb') as f:
                final_results = pickle.load(f)
            scores = calculate_composite_scores(final_results)
            generate_comparison_excel(final_results, scores, XLS_FILE)
            print_text_report(final_results, scores)
            return
        else:
            DATA_MODE = 'RUN_NEW'

    if DATA_MODE == 'RUN_NEW':
        if not os.path.exists(SCENARIO_FILE): return
        with open(SCENARIO_FILE, 'rb') as f:
            scenarios = pickle.load(f)

        env = AirGroundEnv()
        print(f"\n=== Mode: {COMPARISON_MODE} | Number of scenarios: {len(scenarios)} ===")

        for name, cfg in AGENTS_TO_RUN.items():
            if not cfg['run']: continue

            if cfg['type'] in ['HEURISTIC', 'RANDOM']:
                agent = cfg['loader'](env)
            else:
                agent = cfg['loader'](cfg['ep'])

            if agent:
                stats = run_evaluation(agent, env, EVAL_EPISODES, name, cfg['type'], scenarios)
                final_results[name] = stats
            else:
                print(f"Failed to load: {name}")

        if final_results:
            with open(PKL_FILE, 'wb') as f: pickle.dump(final_results, f)
            scores = calculate_composite_scores(final_results)
            generate_comparison_excel(final_results, scores, XLS_FILE)
            print_text_report(final_results, scores)


if __name__ == '__main__':
    main()