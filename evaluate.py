# evaluate.py
import torch, numpy as np, time
from environment.env import AirGroundEnv
from marl.mappo_agent import MAPPOAgent
import config

POWER_LEVELS_dBm = np.linspace(0, config.NODE_TRANSMIT_POWER_dBm, config.ACTION_POWER_LEVELS)
UAV_MOVE_ACTIONS = {
    0: [0, 0, 0], 1: [0, 5, 0], 2: [0, -5, 0], 3: [5, 0, 0], 4: [-5, 0, 0], 5: [0, 0, 5], 6: [0, 0, -5],
}
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

def evaluate():
    MODEL_EPISODE_TO_LOAD = 1400
    EVAL_EPISODES = 50

    env = AirGroundEnv()
    agent = MAPPOAgent(env.num_agents, UAV_ACTION_DIM, UGV_ACTION_DIM)

    try:
        actor_path = f'./models/actor_{MODEL_EPISODE_TO_LOAD}.pth'
        encoder_path = f'./models/encoder_{MODEL_EPISODE_TO_LOAD}.pth'
        agent.load_models(actor_path, encoder_path)
        agent.actor.eval();
        agent.encoder.eval()
        print("--- Model loaded successfully ---")
    except FileNotFoundError:
        print(f"Error: Model file not found.")
        return

    all_rewards = []

    all_reward_components = []

    for ep in range(EVAL_EPISODES):
        obs, state = env.reset()
        hidden_states = [torch.zeros(1, config.RNN_HIDDEN_DIM) for _ in range(env.num_agents)]

        ep_reward = 0
        ep_reward_components = {'throughput': 0, 'connectivity': 0, 'energy_cost': 0}

        for step in range(config.EPISODE_LENGTH):
            avail_actions = env.get_avail_actions()
            actions, _, hs_new = agent.choose_action(obs, hidden_states, avail_actions, deterministic=True)

            for i in range(env.num_uavs, env.num_agents): actions[i] %= UGV_ACTION_DIM

            env_actions = {n.id: map_action_id_to_env_action(i, actions[i], env.num_uavs) for i, n in enumerate(env.nodes)}

            obs, state, reward, done, info = env.step(env_actions)

            hidden_states = hs_new
            ep_reward += reward
            for key, value in info.items():
                if key in ep_reward_components:
                    ep_reward_components[key] += value

            if done: break

        all_rewards.append(ep_reward)
        all_reward_components.append(ep_reward_components)
        print(f"Evaluation Round: {ep + 1}/{EVAL_EPISODES}, This Round's Reward: {ep_reward:.3f}")

    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)

    mean_throughput = np.mean([c['throughput'] for c in all_reward_components])
    mean_connectivity = np.mean([c['connectivity'] for c in all_reward_components])
    mean_energy_cost = np.mean([c['energy_cost'] for c in all_reward_components])

    print(f"\n--- Evaluation completed ---")
    print(f"Model {MODEL_EPISODE_TO_LOAD}'s performance in the {EVAL_EPISODES} evaluation")
    print(f"Average Total Reward: {mean_reward:.3f} +/- {std_reward:.3f}")
    print("--- Subitem Average Score ---")
    print(f"  - Throughput: {mean_throughput:.3f}")
    print(f"  - Connectivity: {mean_connectivity:.3f}")
    print(f"  - Energy Cost: {mean_energy_cost:.3f}")


if __name__ == '__main__':
    evaluate()