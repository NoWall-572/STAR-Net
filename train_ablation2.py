
# train_ablation2.py
# Training script for Ablation 2: ST-GCN-MAPPO

import torch
import numpy as np
import time
import multiprocessing as mp
from environment.env import AirGroundEnv
from marl.mappo_agent_ablation2 import MAPPOAgentAblation2
from marl.buffer import OnPolicyBuffer
from logger import Logger
import config
import os
from reward_normalizer import RewardNormalizer

POWER_LEVELS_dBm = np.linspace(0, config.NODE_TRANSMIT_POWER_dBm, config.ACTION_POWER_LEVELS)
UAV_MOVE_ACTIONS = {
    0: [0, 0, 0], 1: [0, 5, 0], 2: [0, -5, 0], 3: [5, 0, 0], 4: [-5, 0, 0], 5: [0, 0, 5], 6: [0, 0, -5],
}
UGV_MOVE_ACTIONS = {
    0: [0, 0, 0], 1: [0, 3, 0], 2: [0, -3, 0], 3: [3, 0, 0], 4: [-3, 0, 0],
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


def run_episode_batch_worker(agent_state_dict, num_episodes_per_worker):
    try:
        env = AirGroundEnv()
        worker_agent = MAPPOAgentAblation2(
            num_agents=env.num_agents,
            action_dim_uav=UAV_ACTION_DIM,
            action_dim_ugv=UGV_ACTION_DIM,
            device='cpu'
        )
        worker_agent.encoder.load_state_dict(agent_state_dict['encoder'])
        worker_agent.actor.load_state_dict(agent_state_dict['actor'])
        worker_agent.encoder.eval()
        worker_agent.actor.eval()

        local_buffer = []
        episode_infos = []

        for _ in range(num_episodes_per_worker):
            obs, state = env.reset()
            initial_hidden_states_torch = worker_agent.get_initial_hidden_states(obs)
            hidden_states_np = [h.numpy() for h in initial_hidden_states_torch]
            episode_components_raw = {'throughput': 0, 'connectivity': 0, 'energy_cost': 0, 'steps': 0}

            for step in range(config.EPISODE_LENGTH):
                avail_actions = env.get_avail_actions()
                hidden_states_torch = [torch.from_numpy(h) for h in hidden_states_np]
                actions, action_log_probs, new_hidden_states_torch = worker_agent.choose_action(obs, hidden_states_torch, avail_actions)
                new_hidden_states_np = [h.numpy() for h in new_hidden_states_torch]

                for i in range(env.num_uavs, env.num_agents):
                    actions[i] = actions[i] % UGV_ACTION_DIM

                env_actions = {node.id: map_action_id_to_env_action(i, actions[i], env.num_uavs) for i, node in enumerate(env.nodes)}
                next_obs, next_state, _, done, info = env.step(env_actions)

                local_buffer.append(
                    (obs, hidden_states_np, actions, action_log_probs, info, done, state, avail_actions))
                obs, state, hidden_states_np = next_obs, next_state, new_hidden_states_np

                for key, value in info.items():
                    if key in episode_components_raw: episode_components_raw[key] += value
                episode_components_raw['steps'] += 1

                if done: break
            episode_infos.append(episode_components_raw)

        return local_buffer, episode_infos
    except Exception as e:
        print(f"!!! Worker Error: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def main():
    print(f"--- [Ablation 2] Training will run on the device: {config.DEVICE} ---")

    NUM_WORKERS = 8
    EPISODES_PER_WORKER = 3

    env = AirGroundEnv()
    logger = Logger(log_dir='logs_ablation2')
    buffer = OnPolicyBuffer()

    reward_normalizer = RewardNormalizer(num_rewards=3)
    weights = config.CURRENT_REWARD_WEIGHTS
    reward_weights = np.array([weights["throughput"], weights["connectivity"], weights["energy"]])

    agent = MAPPOAgentAblation2(
        num_agents=env.num_agents,
        action_dim_uav=UAV_ACTION_DIM,
        action_dim_ugv=UGV_ACTION_DIM
    )
    total_episodes, total_steps = 0, 0

    print("--- Begin training [Ablation 2: ST-GCN-MAPPO] ---")
    print(f"--- Using {NUM_WORKERS} parallel worker processes ---")

    for update_cycle in range(1, 3001):
        start_time = time.time()
        agent_state_dict = {
            'encoder': {k: v.cpu() for k, v in agent.encoder.state_dict().items()},
            'actor': {k: v.cpu() for k, v in agent.actor.state_dict().items()}
        }

        print(f"\n[Update cycle {update_cycle}] Starting parallel data collection...")
        with mp.Pool(processes=NUM_WORKERS) as pool:
            results = pool.starmap_async(run_episode_batch_worker, [(agent_state_dict, EPISODES_PER_WORKER) for _ in range(NUM_WORKERS)])
            worker_outputs = results.get()

        current_batch_steps = 0
        all_episode_infos = []
        for local_buffer, episode_infos in worker_outputs:
            for experience in local_buffer:
                obs, hidden_np, act, log_p, info, done, state, avail = experience
                hidden_torch = [torch.from_numpy(h) for h in hidden_np]
                raw_reward_components = np.array(
                    [info.get('throughput', 0), info.get('connectivity', 0), info.get('energy_cost', 0)])
                normalized_rewards = reward_normalizer.normalize_step(raw_reward_components)
                final_reward = np.dot(reward_weights, normalized_rewards)
                buffer.push(obs, hidden_torch, act, log_p, final_reward, done, state, avail)
                current_batch_steps += 1
            all_episode_infos.extend(episode_infos)

        if current_batch_steps == 0:
            print("!!! Warning: None of the workers collected any data; skipping this update cycle.")
            continue

        total_steps += current_batch_steps
        total_episodes += len(all_episode_infos)
        reward_normalizer.update_after_episode()

        collection_time = time.time() - start_time
        print(f"Data collection completed. Time taken: {collection_time:.2f}s. Total {len(all_episode_infos)} rounds collected, {current_batch_steps} steps.")

        print("Starting model update...")
        start_update_time = time.time()
        diagnostics = agent.update(buffer)
        buffer.clear()
        update_time = time.time() - start_update_time
        print(f"Model update completed. Duration: {update_time:.2f}s. (Critic Loss: {diagnostics['critic_loss']:.4f}, Exp_Var: {diagnostics['explained_variance']:.3f})")

        avg_throughput = np.mean([info['throughput'] for info in all_episode_infos])
        avg_connectivity = np.mean([info['connectivity'] for info in all_episode_infos])
        avg_energy = np.mean([info['energy_cost'] for info in all_episode_infos])
        logger.log(total_episodes, total_steps, {'throughput': avg_throughput, 'connectivity': avg_connectivity, 'energy_cost': avg_energy}, diagnostics)

        print(
            f"Batch Average -> Total Rounds: {total_episodes} | Throughput: {avg_throughput:.2f} | Connectivity: {avg_connectivity:.4f} | Energy Consumption: {avg_energy:.4f}")

        if update_cycle % 10 == 0:
            agent.save_models('./models_ablation2', total_episodes)
            print(f"--- [Ablation 2] Save the model at epoch {total_episodes} ---")

    logger.close()


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

