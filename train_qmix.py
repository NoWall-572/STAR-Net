# train_qmix.py
import torch, numpy as np, time
from collections import defaultdict
from environment.env import AirGroundEnv
from logger import Logger
from marl.qmix_agent import QMIXAgent
from marl.replay_buffer import EpisodeReplayBuffer

try:
    import config_qmix as config

    print("--- Loaded QMIX config ---")
except ImportError:
    import config

    print("--- WARNING: QMIX config not found, using default ---")

POWER_LEVELS_dBm = np.linspace(0, config.NODE_TRANSMIT_POWER_dBm, config.ACTION_POWER_LEVELS)
UAV_MOVE_ACTIONS = {0: [0, 0, 0], 1: [0, 5, 0], 2: [0, -5, 0], 3: [5, 0, 0], 4: [-5, 0, 0], 5: [0, 0, 5], 6: [0, 0, -5]}
UGV_MOVE_ACTIONS = {0: [0, 0, 0], 1: [0, 3, 0], 2: [0, -3, 0], 3: [3, 0, 0], 4: [-3, 0, 0]}
UAV_ACTION_DIM = len(UAV_MOVE_ACTIONS) * config.ACTION_POWER_LEVELS
UGV_ACTION_DIM = len(UGV_MOVE_ACTIONS) * config.ACTION_POWER_LEVELS


def map_action(agent_id, action, num_uavs):
    is_uav = agent_id < num_uavs
    action_dim = UAV_ACTION_DIM if is_uav else UGV_ACTION_DIM
    moves = UAV_MOVE_ACTIONS if is_uav else UGV_MOVE_ACTIONS
    action %= action_dim
    p_idx, m_idx = action % config.ACTION_POWER_LEVELS, action // config.ACTION_POWER_LEVELS
    accel = moves.get(m_idx, [0, 0, 0])
    watts = 10 ** ((POWER_LEVELS_dBm[p_idx] - 30) / 10)
    return {'acceleration': accel, 'transmit_power_watts': watts}


def process_obs(obs_list):
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
    return np.array(processed, dtype=np.float32)


def main():
    env = AirGroundEnv()
    logger = Logger(log_dir='logs_qmix')
    obs_dim = config.MAX_OBS_NODES * config.NODE_FEATURE_DIM
    state_dim = env.num_agents * obs_dim
    action_dim = UAV_ACTION_DIM
    agent = QMIXAgent(env.num_agents, action_dim, state_dim, obs_dim)
    buffer = EpisodeReplayBuffer(config.BUFFER_CAPACITY, config.EPISODE_LENGTH, state_dim, obs_dim,
                                 action_dim, env.num_agents, config.RNN_HIDDEN_DIM, agent.device)
    total_steps, epsilon = 0, config.EPSILON_START
    print(f"--- Starting QMIX Training on {agent.device} ---")

    for episode in range(1, config.TOTAL_EPISODES + 1):
        ep_data = defaultdict(list)
        obs, _ = env.reset()
        h = torch.zeros(env.num_agents, config.RNN_HIDDEN_DIM).to(agent.device)
        ep_reward, done = 0, False

        for t in range(config.EPISODE_LENGTH):
            if done: break
            ep_data['hidden_state'].append(h.cpu().numpy())
            flat_obs = process_obs(obs)
            avail = env.get_avail_actions()

            actions = []
            with torch.no_grad():
                for i in range(env.num_agents):
                    if np.random.rand() < epsilon:
                        a = np.random.choice(np.nonzero(avail[i])[0]) if np.any(avail[i]) else 0
                    else:
                        q, new_h = agent.agent_net(torch.FloatTensor(flat_obs[i]).unsqueeze(0).to(agent.device),
                                                   h[i].unsqueeze(0))
                        h[i] = new_h.squeeze(0)
                        q[torch.FloatTensor(avail[i]).unsqueeze(0).to(agent.device) == 0] = -float('inf')
                        a = q.argmax().item()
                    actions.append(a)

            env_actions = {n.id: map_action(i, a, env.num_uavs) for i, (n, a) in enumerate(zip(env.nodes, actions))}
            next_obs, _, reward, done, _ = env.step(env_actions)

            ep_data['state'].append(flat_obs.flatten())
            ep_data['obs'].append(flat_obs)
            ep_data['action'].append(actions)
            ep_data['reward'].append(reward)
            ep_data['done'].append(done)
            ep_data['avail_actions'].append(avail)

            obs = next_obs
            ep_reward += reward
            total_steps += 1
            epsilon = max(config.EPSILON_FINISH, epsilon - (config.EPSILON_START - config.EPSILON_FINISH) / config.EPSILON_DECAY_STEPS)

        buffer.store_episode({k: np.array(v) for k, v in ep_data.items()})

        if len(buffer) >= config.BATCH_SIZE:
            for _ in range(config.TRAIN_STEPS_PER_EPISODE):
                agent.train(buffer.sample(config.BATCH_SIZE))

        if episode % config.TARGET_UPDATE_FREQ == 0: agent.update_targets()
        if episode % 20 == 0: print(
            f"Ep: {episode:>5} | Steps: {total_steps:>7} | Reward: {ep_reward:8.2f} | Eps: {epsilon:.3f}")
        if episode % config.SAVE_INTERVAL == 0: agent.save('./models_qmix', episode)

    logger.close()


if __name__ == '__main__':
    main()