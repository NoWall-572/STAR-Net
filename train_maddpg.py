# train_maddpg.py

import torch
import numpy as np
import time
from environment.env import AirGroundEnv
from marl.maddpg_agent import MADDPGAgent
from marl.buffer_maddpg import ReplayBuffer
from logger import Logger
import config
import os

print(f"--- MADDPG training will run on device: {config.DEVICE} ---")

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
        accel = UAV_MOVE_ACTIONS.get(move_idx, [0, 0, 0])
    else:
        power_idx = action_id % config.ACTION_POWER_LEVELS
        move_idx = action_id // config.ACTION_POWER_LEVELS
        accel = UGV_MOVE_ACTIONS.get(move_idx, [0, 0, 0])
    transmit_power_watts = 10 ** ((POWER_LEVELS_dBm[power_idx] - 30) / 10)
    return {'acceleration': accel, 'transmit_power_watts': transmit_power_watts}


def main():
    TOTAL_STEPS = 500000
    BUFFER_CAPACITY = 100000
    BATCH_SIZE = 256
    EXPLORATION_START = 1.0
    EXPLORATION_END = 0.05
    EXPLORATION_DECAY = (TOTAL_STEPS - 10000) / 1.5
    SAVE_INTERVAL = 5000

    env = AirGroundEnv()
    logger = Logger()
    buffer = ReplayBuffer(BUFFER_CAPACITY, env.num_agents)

    agent = MADDPGAgent(
        num_agents=env.num_agents,
        action_dim_uav=UAV_ACTION_DIM,
        action_dim_ugv=UGV_ACTION_DIM,
        num_uavs=env.num_uavs
    )

    print("--- Begin Training (MADDPG) ---")

    total_episodes = 0
    obs, state = env.reset()
    hidden_states = [torch.zeros(1, config.RNN_HIDDEN_DIM) for _ in range(env.num_agents)]

    current_episode_reward = 0
    ep_reward_components = {'throughput': 0, 'connectivity': 0, 'energy_cost': 0}

    for step in range(TOTAL_STEPS):
        avail_actions = env.get_avail_actions()

        epsilon = max(EXPLORATION_END, EXPLORATION_START - (step / EXPLORATION_DECAY))
        explore_decision = np.random.rand() < epsilon

        actions, new_hidden_states = agent.choose_action(obs, hidden_states, avail_actions, explore=explore_decision)

        env_actions = {node.id: map_action_id_to_env_action(i, actions[i], env.num_uavs) for i, node in enumerate(env.nodes)}

        next_obs, next_state, reward, done, info = env.step(env_actions)

        current_episode_reward += reward
        for key, value in info.items():
            if key in ep_reward_components:
                ep_reward_components[key] += value

        next_avail_actions = env.get_avail_actions()
        buffer.push(obs, hidden_states, actions, reward, next_obs, new_hidden_states, done, avail_actions, next_avail_actions)

        obs = next_obs
        hidden_states = new_hidden_states

        diagnostics = agent.update(buffer, BATCH_SIZE)

        if done:
            total_episodes += 1
            print(
                f"Episode: {total_episodes} | Steps: {step + 1}/{TOTAL_STEPS} | Episode Reward: {current_episode_reward:.3f} | Epsilon: {epsilon:.3f}")
            if diagnostics:
                logger.log(total_episodes, step + 1, ep_reward_components, diagnostics)

            obs, state = env.reset()
            hidden_states = [torch.zeros(1, config.RNN_HIDDEN_DIM) for _ in range(env.num_agents)]
            current_episode_reward = 0
            ep_reward_components = {'throughput': 0, 'connectivity': 0, 'energy_cost': 0}

        # 定期保存模型
        if (step + 1) % SAVE_INTERVAL == 0:
            agent.save_models('./models_maddpg', step + 1)
            print(f"--- Save the model at step {step + 1} ---")

    logger.close()


if __name__ == '__main__':
    main()