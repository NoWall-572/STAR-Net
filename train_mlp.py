# train_mlp.py

import torch
import numpy as np
import time
import multiprocessing as mp
from environment.env import AirGroundEnv
from marl.mappo_agent_mlp import MAPPOAgentMLP
from marl.buffer import OnPolicyBuffer
from logger import Logger
import config
import os
from reward_normalizer import RewardNormalizer

POWER_LEVELS_dBm = np.linspace(0, config.NODE_TRANSMIT_POWER_dBm, config.ACTION_POWER_LEVELS)
UAV_MOVE_ACTIONS = {0: [0, 0, 0], 1: [0, 5, 0], 2: [0, -5, 0], 3: [5, 0, 0], 4: [-5, 0, 0], 5: [0, 0, 5], 6: [0, 0, -5]}
UGV_MOVE_ACTIONS = {
    0: [0, 0, 0],
    1: [0, 3, 0],
    2: [0, -3, 0],
    3: [3, 0, 0],
    4: [-3, 0, 0],
}
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

def data_collection_worker_mlp(agent_state_dict, queue, num_episodes_to_run, obs_dim):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    env = AirGroundEnv()
    worker_agent = MAPPOAgentMLP(
        num_agents=env.num_agents,
        action_dim_uav=UAV_ACTION_DIM,
        action_dim_ugv=UGV_ACTION_DIM,
        obs_dim=obs_dim,
        device='cpu'
    )
    worker_agent.encoder.load_state_dict(agent_state_dict['encoder'])
    worker_agent.actor.load_state_dict(agent_state_dict['actor'])
    worker_agent.encoder.eval();
    worker_agent.actor.eval()

    def pad_and_flatten_obs(obs_list):
        padded_obs_vectors = []
        for o in obs_list:
            features = o['node_features']

            num_nodes = features.shape[0]
            feature_dim = config.NODE_FEATURE_DIM
            # ++++++++++++++++++++++++++++++++++++++++++++++

            padded_features = np.zeros((config.MAX_OBS_NODES, feature_dim), dtype=np.float32)
            if num_nodes > 0:
                len_to_copy = min(num_nodes, config.MAX_OBS_NODES)
                len_feat_to_copy = min(features.shape[1], feature_dim)
                padded_features[:len_to_copy, :len_feat_to_copy] = features[:len_to_copy, :len_feat_to_copy]
                # +++++++++++++++++++++++++++++++++++++++++++

            padded_obs_vectors.append(padded_features.flatten())
        return padded_obs_vectors

    for _ in range(num_episodes_to_run):
        obs, state = env.reset()
        flat_obs = pad_and_flatten_obs(obs)
        hidden_states_np = [np.zeros((1, config.RNN_HIDDEN_DIM), dtype=np.float32) for _ in range(env.num_agents)]
        episode_data = [];
        episode_info = {'throughput': 0, 'connectivity': 0, 'energy_cost': 0}

        for step in range(config.EPISODE_LENGTH):
            avail_actions = env.get_avail_actions()
            hidden_states_torch = [torch.from_numpy(h) for h in hidden_states_np]

            actions = []
            action_log_probs = []
            new_hidden_states_torch = []

            for i in range(env.num_agents):
                action, log_prob, new_hs = worker_agent.choose_action(
                    flat_obs[i],
                    hidden_states_torch[i],
                    avail_actions[i]
                )
                actions.append(action)
                action_log_probs.append(log_prob)
                new_hidden_states_torch.append(new_hs)

            new_hidden_states_np = [h.numpy() for h in new_hidden_states_torch]

            for i in range(env.num_uavs, env.num_agents): actions[i] %= UGV_ACTION_DIM
            env_actions = {node.id: map_action_id_to_env_action(i, actions[i], env.num_uavs) for i, node in enumerate(env.nodes)}
            next_obs, next_state, _, done, info = env.step(env_actions)

            next_flat_obs = pad_and_flatten_obs(next_obs)
            episode_data.append(
                (flat_obs, hidden_states_np, actions, action_log_probs, info, done, state, avail_actions))

            flat_obs = next_flat_obs
            hidden_states_np = new_hidden_states_np

            for key, value in info.items():
                if key in episode_info: episode_info[key] += value
            if done: break

        queue.put((episode_data, episode_info))


def main():
    NUM_WORKERS = 8
    EPISODES_PER_UPDATE = 64

    env = AirGroundEnv()

    obs_dim = config.MAX_OBS_NODES * config.NODE_FEATURE_DIM

    logger = Logger(log_dir='logs_mlp')
    buffer = OnPolicyBuffer()
    reward_normalizer = RewardNormalizer(num_rewards=3)
    weights = config.CURRENT_REWARD_WEIGHTS
    reward_weights = np.array([weights["throughput"], weights["connectivity"], weights["energy"]])

    agent = MAPPOAgentMLP(
        num_agents=env.num_agents,
        action_dim_uav=UAV_ACTION_DIM,
        action_dim_ugv=UGV_ACTION_DIM,
        obs_dim=obs_dim
    )

    total_steps = 0;
    total_episodes = 0
    queue = mp.Queue()
    episodes_per_worker = EPISODES_PER_UPDATE // NUM_WORKERS
    remaining_episodes = EPISODES_PER_UPDATE % NUM_WORKERS

    for update_cycle in range(1, 1000):
        start_time = time.time()
        agent_state_dict = {
            'encoder': {k: v.cpu() for k, v in agent.encoder.state_dict().items()},
            'actor': {k: v.cpu() for k, v in agent.actor.state_dict().items()}
        }

        processes = []
        for i in range(NUM_WORKERS):
            eps_to_run = episodes_per_worker + (1 if i < remaining_episodes else 0)
            if eps_to_run > 0:
                p = mp.Process(target=data_collection_worker_mlp, args=(agent_state_dict, queue, eps_to_run, obs_dim))
                processes.append(p);
                p.start()

        all_episode_infos = []
        for _ in range(EPISODES_PER_UPDATE):
            episode_data, episode_info = queue.get()
            all_episode_infos.append(episode_info)
            for experience in episode_data:
                flat_obs, hidden_np, act, log_p, info, done, state, avail = experience
                hidden_torch = [torch.from_numpy(h) for h in hidden_np]
                raw_reward_components = np.array(
                    [info.get('throughput', 0), info.get('connectivity', 0), info.get('energy_cost', 0)])
                normalized_rewards = reward_normalizer.normalize_step(raw_reward_components)
                final_reward_for_agent = np.dot(reward_weights, normalized_rewards)
                buffer.push(flat_obs, hidden_torch, act, log_p, final_reward_for_agent, done, state, avail)

        for p in processes: p.join()

        total_steps += len(buffer.rewards);
        total_episodes += EPISODES_PER_UPDATE
        reward_normalizer.update_after_episode()
        print(f"Data collection completed. Duration: {time.time() - start_time:.2f}s.")

        start_update_time = time.time()
        diagnostics = agent.update(buffer)
        buffer.clear()
        print(f"Model update completed. Duration: {time.time() - start_update_time:.2f}s.")

        avg_throughput = np.mean([info['throughput'] for info in all_episode_infos])
        avg_connectivity = np.mean([info['connectivity'] for info in all_episode_infos])
        avg_energy = np.mean([info['energy_cost'] for info in all_episode_infos])
        logger.log(total_episodes, total_steps, {'throughput': avg_throughput, 'connectivity': avg_connectivity, 'energy_cost': avg_energy}, diagnostics)

        print(
            f"Batch Average -> Total Rounds: {total_episodes} | Total Steps: {total_steps} | "
            f"Throughput: {avg_throughput:7.2f} | "
            f"Overall efficiency: {avg_connectivity:6.4f} | "
            f"Energy consumption: {avg_energy:7.2f}"
        )

        SAVE_INTERVAL = 10
        if update_cycle % SAVE_INTERVAL == 0:
            agent.save_models('./models_mlp', total_episodes)
            print(f"--- Save the MLP model at the {total_episodes}th iteration ---")

    logger.close()


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()