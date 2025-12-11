# run_node_attack.py

import torch
import numpy as np
import networkx as nx
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from environment.env import AirGroundEnv
import config
from environment.entities import UAV, UGV
from environment import channel_models

from marl.mappo_agent import MAPPOAgent
from marl.mappo_agent_mlp import MAPPOAgentMLP
from marl.ippo_agent import IPPOAgent
from marl.qmix_agent import QMIXAgent
from marl.maddpg_agent import MADDPGAgent
from baselines import RandomAgent, HeuristicAgentV4_Expert
from marl.mappo_agent_ablation1 import MAPPOAgentAblation1
from marl.mappo_agent_ablation2 import MAPPOAgentAblation2
from marl.mappo_agent_ablation4 import MAPPOAgentAblation4

UAV_MOVE_ACTIONS = {0: [0, 0, 0], 1: [0, 5, 0], 2: [0, -5, 0], 3: [5, 0, 0], 4: [-5, 0, 0], 5: [0, 0, 5], 6: [0, 0, -5]}
UGV_MOVE_ACTIONS = {0: [0, 0, 0], 1: [0, 3, 0], 2: [0, -3, 0], 3: [3, 0, 0], 4: [-3, 0, 0]}
POWER_LEVELS_dBm = np.linspace(0, config.NODE_TRANSMIT_POWER_dBm, config.ACTION_POWER_LEVELS)
UAV_ACTION_DIM = len(UAV_MOVE_ACTIONS) * config.ACTION_POWER_LEVELS
UGV_ACTION_DIM = len(UGV_MOVE_ACTIONS) * config.ACTION_POWER_LEVELS


def map_action_id_to_env_action(agent_id, action_id, num_uavs):
    if agent_id < num_uavs:
        power_idx = action_id % config.ACTION_POWER_LEVELS;
        move_idx = action_id // config.ACTION_POWER_LEVELS
        accel = UAV_MOVE_ACTIONS[move_idx]
    else:
        power_idx = action_id % config.ACTION_POWER_LEVELS;
        move_idx = action_id // config.ACTION_POWER_LEVELS
        accel = UGV_MOVE_ACTIONS[move_idx]
    return {'acceleration': accel, 'transmit_power_watts': 10 ** ((POWER_LEVELS_dBm[power_idx] - 30) / 10)}


def process_obs_for_flat_agents(obs_list):
    processed = []
    for o in obs_list:
        feat = o.get('node_features', np.array([]))
        padded = np.zeros((config.MAX_OBS_NODES, config.NODE_FEATURE_DIM), dtype=np.float32)
        if feat.size > 0:
            n, f = feat.shape;
            padded[:min(n, config.MAX_OBS_NODES), :min(f, config.NODE_FEATURE_DIM)] = feat[
                :min(n, config.MAX_OBS_NODES), :min(f, config.NODE_FEATURE_DIM)]
        processed.append(padded.flatten())
    return processed


def process_obs_for_ablation5(obs_list):
    processed = []
    for o in obs_list:
        processed.append(
            {'edge_index': o['edge_index'], 'edge_attr': o['edge_attr'], 'node_features': o['node_features'][:, :-2]})
    return processed


def calculate_surbi_ranking(G, r=0.3):
    try:
        return nx.degree_centrality(G)
    except:
        return {n: 0 for n in G.nodes()}


def load_stgat(path, ep):
    a = MAPPOAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_mlp(path, ep):
    obs = config.MAX_OBS_NODES * config.NODE_FEATURE_DIM
    a = MAPPOAgentMLP(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM, obs)
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_qmix(path, ep):
    obs = config.MAX_OBS_NODES * config.NODE_FEATURE_DIM;
    state = (config.NUM_UAVS + config.NUM_UGVS) * obs
    a = QMIXAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, state, obs)
    try:
        a.load(path, ep); return a
    except:
        return None


def load_ippo(path, ep):
    a = IPPOAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    try:
        a.load_models(f'{path}/actor_ippo_{ep}.pth', f'{path}/critic_ippo_{ep}.pth',
                      f'{path}/encoder_ippo_{ep}.pth'); return a
    except:
        return None


def load_maddpg(path, ep):
    a = MADDPGAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM, config.NUM_UAVS)
    try:
        a.load_models(path, ep); return a
    except:
        return None


def load_abl1(path, ep):
    a = MAPPOAgentAblation1(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_abl2(path, ep):
    a = MAPPOAgentAblation2(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_abl4(path, ep):
    a = MAPPOAgentAblation4(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM)
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_abl5(path, ep):
    od = config.NODE_FEATURE_DIM;
    config.NODE_FEATURE_DIM = od - 2
    a = MAPPOAgent(config.NUM_UAVS + config.NUM_UGVS, UAV_ACTION_DIM, UGV_ACTION_DIM);
    config.NODE_FEATURE_DIM = od
    try:
        a.load_models(f'{path}/actor_{ep}.pth', f'{path}/encoder_{ep}.pth'); return a
    except:
        return None


def load_heur(env): return HeuristicAgentV4_Expert(env.num_agents, env.nodes, config.ACTION_POWER_LEVELS,
                                                   UAV_MOVE_ACTIONS, env.num_uavs, env.num_ugvs, env.area_size)


def load_rand(env): return RandomAgent(env.num_agents, UAV_ACTION_DIM, UGV_ACTION_DIM, env.num_uavs)


def plot_side_by_side_metrics(plot_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
    x = np.arange(3);
    x_labels = ["Pre-Attack", "Under Attack", "Recovered"]
    colors = plt.get_cmap('tab10', len(plot_data));
    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']

    handles, labels = [], []
    for i, (name, m) in enumerate(plot_data.items()):
        c, mk = colors(i), markers[i % len(markers)]
        l1, = ax1.plot(x, m['throughput'], color=c, linestyle='-', linewidth=2.5, marker=mk, markersize=9, label=name)
        ax2.plot(x, m['connectivity'], color=c, linestyle='-', linewidth=2.5, marker=mk, markersize=9)
        handles.append(l1);
        labels.append(name)

    for ax, title, ylabel in zip([ax1, ax2], ['Throughput Performance', 'Topological Robustness'],
                                 ['Throughput (Mbps)', 'Natural Connectivity']):
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold');
        ax.set_xlabel('Attack Stage', fontsize=14, fontweight='bold')
        ax.set_xticks(x);
        ax.set_xticklabels(x_labels, fontsize=12);
        ax.grid(True, linestyle=':', alpha=0.7)
        ymin, ymax = ax.get_ylim();
        ax.set_ylim(0, ymax * 1.1)

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=11, fancybox=True,
               shadow=True)
    plt.tight_layout();
    plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.2);
    plt.suptitle('Network Resilience Analysis', fontsize=20, y=0.96);
    plt.show()


def generate_excel_report(all_results, filename):
    print(f"--- Generate reports: {filename} ---")
    row_labels = ["Throughput (Pre)", "Throughput (Damage)", "Throughput (Recovered)",
                  "Connectivity (Pre)", "Connectivity (Damage)", "Connectivity (Recovered)"]
    data_dict = {"Metrics": row_labels}

    for name, metrics in all_results.items():
        thr = metrics['throughput'];
        conn = metrics['connectivity']
        data_dict[name] = [round(thr[0], 4), round(thr[1], 4), round(thr[2], 4), round(conn[0], 4), round(conn[1], 4),
                           round(conn[2], 4)]

    df = pd.DataFrame(data_dict)
    try:
        df.to_excel(filename, index=False); print(f"Excel saved successfully: {filename}")
    except:
        df.to_csv(filename.replace('.xlsx', '.csv'), index=False); print("Excel save failed. Saved as CSV.")


def main():
    DATA_MODE = 'RUN_NEW'
    # 'BASELINE' or 'ABLATION'
    COMPARISON_MODE = 'ABLATION'

    if COMPARISON_MODE == 'BASELINE':
        PKL_FILE = 'attack_baseline.pkl';
        XLS_FILE = 'attack_baseline.xlsx'
    else:
        PKL_FILE = 'attack_ablation.pkl';
        XLS_FILE = 'attack_ablation.xlsx'

    SCENARIO_FILE = 'evaluation_scenarios.pkl'

    AGENTS_CONFIG = {
        "STGAT-MAPPO (Ours)": {"run": True, "loader": load_stgat, "path": "models_env_2_2_6", "ep": 6720,
                               "type": "STGAT"},

        "MAPPO-MLP": {"run": False, "loader": load_mlp, "path": "models_mlp_env_2_2_6", "ep": 9600, "type": "MLP"},
        "QMIX-STGAT": {"run": False, "loader": load_qmix, "path": "models_qmix_env_2_2_6", "ep": 1500, "type": "QMIX"},
        "IPPO-STGAT": {"run": False, "loader": load_ippo, "path": "models_ippo_env_2_2_6", "ep": 6720, "type": "IPPO"},
        "MADDPG-STGAT": {"run": False, "loader": load_maddpg, "path": "models_maddpg_env_2_2_6", "ep": 45000,
                         "type": "MADDPG"},
        "Heuristic": {"run": False, "loader": load_heur, "path": None, "ep": None, "type": "HEURISTIC"},
        "Random": {"run": False, "loader": load_rand, "path": None, "ep": None, "type": "RANDOM"},

        "S-GAT-MAPPO": {"run": True, "loader": load_abl1, "path": "models_ablation1_env_2_2_6", "ep": 2400,
                        "type": "ABLATION1"},
        "ST-GCN-MAPPO": {"run": True, "loader": load_abl2, "path": "models_ablation2_env_2_2_6", "ep": 1680,
                         "type": "ABLATION2"},
        "S-GCN-MAPPO": {"run": True, "loader": load_abl4, "path": "models_ablation4_env_2_2_6", "ep": 2640,
                        "type": "ABLATION4"},
        "STGAT-MAPPO (No Hetero)": {"run": True, "loader": load_abl5, "path": "models_ablation5_env_2_2_6", "ep": 3600,
                                    "type": "ABLATION5"},
    }
    # ========================================

    if DATA_MODE == 'LOAD_ONLY' and os.path.exists(PKL_FILE):
        with open(PKL_FILE, 'rb') as f: all_results = pickle.load(f)
        generate_excel_report(all_results, XLS_FILE);
        plot_side_by_side_metrics(all_results);
        return

    if DATA_MODE == 'RUN_NEW':
        try:
            with open(SCENARIO_FILE, 'rb') as f:
                scenarios = pickle.load(f); test_scenario = scenarios[1]
        except:
            print("Scene loading failed"); return

        if COMPARISON_MODE == 'BASELINE':
            target_groups = ["STGAT", "MLP", "QMIX", "IPPO", "MADDPG", "HEURISTIC", "RANDOM"]
        else:
            target_groups = ["STGAT", "ABLATION1", "ABLATION2", "ABLATION4", "ABLATION5"]

        all_results = {}
        print(f"\n=== Mode: {COMPARISON_MODE} (Attack Test) ===")

        for name, cfg in AGENTS_CONFIG.items():
            if not cfg['run']: continue
            if cfg['type'] not in target_groups and "STGAT" not in cfg['type']: continue

            env = AirGroundEnv()
            total_initial_energy = config.NUM_UAVS * config.UAV_INITIAL_ENERGY

            print(f"\n--- Test Model: {name} ---")

            agent = cfg['loader'](env) if cfg['type'] in ['HEURISTIC', 'RANDOM'] else cfg['loader'](cfg['path'],
                                                                                                    cfg['ep'])
            if not agent: print(f"Failed to load: {name}"); continue

            obs, _ = env.reset(scenario_config=test_scenario)
            h = [torch.zeros(1, config.RNN_HIDDEN_DIM) for _ in range(env.num_agents)]
            attack_step = config.EPISODE_LENGTH // 2

            is_flat = cfg['type'] in ["MLP", "QMIX"]
            is_abl5 = cfg['type'] == "ABLATION5"
            if is_flat: flat_obs = process_obs_for_flat_agents(obs)
            if is_abl5: obs = process_obs_for_ablation5(obs)

            for _ in range(attack_step):
                avail = env.get_avail_actions()
                if cfg['type'] in ["STGAT", "IPPO", "ABLATION2", "ABLATION5"]:
                    a, _, h = agent.choose_action(obs, h, avail, True)
                elif cfg['type'] in ["ABLATION1", "ABLATION4"]:
                    a, _, _ = agent.choose_action(obs, None, avail, True)
                elif cfg['type'] == "MLP":
                    a, h_new = [], [];
                    for i in range(env.num_agents): act, _, hi = agent.choose_action(flat_obs[i], h[i], avail[i], True); a.append(act); h_new.append(hi)
                    h = h_new
                elif cfg['type'] == "QMIX":
                    a, raw = agent.choose_action(flat_obs, h, avail); h = [i.unsqueeze(0) for i in raw]
                elif cfg['type'] == "MADDPG":
                    a, h = agent.choose_action(obs, h, avail, False)
                elif isinstance(agent, RandomAgent):
                    a = agent.choose_action(avail)
                elif isinstance(agent, HeuristicAgentV4_Expert):
                    agent.nodes = env.nodes; a = agent.choose_action(obs)

                env.step({n.id: map_action_id_to_env_action(i, act, env.num_uavs) for i, (n, act) in
                          enumerate(zip(env.nodes, a))})
                obs = env._get_obs()
                if is_flat: flat_obs = process_obs_for_flat_agents(obs)
                if is_abl5: obs = process_obs_for_ablation5(obs)

            pre = env.get_current_performance_metrics()
            used_energy_pre = total_initial_energy - pre.get('total_remaining_energy', 0)

            try:
                target = max(calculate_surbi_ranking(env.G), key=calculate_surbi_ranking(env.G).get)
            except:
                target = 0

            is_uav = target < config.NUM_UAVS
            env.nodes.pop(target);
            env.uavs = [n for n in env.nodes if isinstance(n, UAV)];
            env.ugvs = [n for n in env.nodes if isinstance(n, UGV)]
            env.num_agents = len(env.nodes);
            env.num_uavs = len(env.uavs);
            env.num_ugvs = len(env.ugvs)
            env.transmit_powers = np.delete(env.transmit_powers, target, 0);
            env.node_id_to_idx = {n.id: i for i, n in enumerate(env.nodes)}
            env.positions = np.array([n.pos for n in env.nodes]);
            env.velocities = np.array([n.vel for n in env.nodes])
            env.max_speeds = np.array([n.max_speed for n in env.nodes]);
            env.max_accelerations = np.array([n.max_acceleration for n in env.nodes])
            env.is_uav_mask = np.array([isinstance(n, UAV) for n in env.nodes])
            if is_uav: env.remaining_energies = np.delete(env.remaining_energies, target, 0)

            if hasattr(agent, 'num_agents'): agent.num_agents = env.num_agents
            if hasattr(agent, 'n_agents'): agent.n_agents = env.num_agents
            if isinstance(agent, (HeuristicAgentV4_Expert, RandomAgent)): agent.num_uavs = env.num_uavs; agent.num_ugvs = env.num_ugvs

            obs.pop(target);
            h.pop(target)
            env._update_network_graph(env.transmit_powers)

            inst = env.get_current_performance_metrics()
            obs = env._get_obs()
            if is_flat: flat_obs = process_obs_for_flat_agents(obs)
            if is_abl5: obs = process_obs_for_ablation5(obs)

            for _ in range(attack_step, config.EPISODE_LENGTH):
                avail = env.get_avail_actions()
                if cfg['type'] in ["STGAT", "IPPO", "ABLATION2", "ABLATION5"]:
                    a, _, h = agent.choose_action(obs, h, avail, True)
                elif cfg['type'] in ["ABLATION1", "ABLATION4"]:
                    a, _, _ = agent.choose_action(obs, None, avail, True)
                elif cfg['type'] == "MLP":
                    a, h_new = [], [];
                    for i in range(env.num_agents): act, _, hi = agent.choose_action(flat_obs[i], h[i], avail[i], True); a.append(act); h_new.append(hi)
                    h = h_new
                elif cfg['type'] == "QMIX":
                    a, raw = agent.choose_action(flat_obs, h, avail); h = [i.unsqueeze(0) for i in raw]
                elif cfg['type'] == "MADDPG":
                    a, h = agent.choose_action(obs, h, avail, False)
                elif isinstance(agent, RandomAgent):
                    a = agent.choose_action(avail)
                elif isinstance(agent, HeuristicAgentV4_Expert):
                    agent.nodes = env.nodes; a = agent.choose_action(obs)

                env.step({n.id: map_action_id_to_env_action(i, act, env.num_uavs) for i, (n, act) in
                          enumerate(zip(env.nodes, a))})
                obs = env._get_obs()
                if is_flat: flat_obs = process_obs_for_flat_agents(obs)
                if is_abl5: obs = process_obs_for_ablation5(obs)

            post = env.get_current_performance_metrics()

            all_results[name] = {
                "throughput": [pre.get('throughput', 0) / 1e6, inst.get('throughput', 0) / 1e6,
                               post.get('throughput', 0) / 1e6],
                "connectivity": [pre.get('connectivity', 0), inst.get('connectivity', 0), post.get('connectivity', 0)]
            }

        if all_results:
            with open(PKL_FILE, 'wb') as f:
                pickle.dump(all_results, f)
            generate_excel_report(all_results, XLS_FILE);
            plot_side_by_side_metrics(all_results)
        else:
            print("No data collected.")


if __name__ == '__main__':
    main()