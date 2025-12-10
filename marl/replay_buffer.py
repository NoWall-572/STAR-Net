# marl/replay_buffer.py
import numpy as np
import torch


class EpisodeReplayBuffer:
    def __init__(self, capacity, ep_limit, state_dim, obs_dim, n_actions, n_agents, hidden_dim, device):
        self.capacity, self.ep_limit, self.device = capacity, ep_limit, device
        self.states = np.zeros((capacity, ep_limit, state_dim), dtype=np.float32)
        self.obs = np.zeros((capacity, ep_limit, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, ep_limit, n_agents, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, ep_limit, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, ep_limit, 1), dtype=np.bool_)
        self.avail_actions = np.zeros((capacity, ep_limit, n_agents, n_actions), dtype=np.bool_)
        self.hidden_states = np.zeros((capacity, ep_limit, n_agents, hidden_dim), dtype=np.float32)
        self.filled = np.zeros((capacity, ep_limit, 1), dtype=np.bool_)
        self.current_idx, self.current_size = 0, 0

    def store_episode(self, ep_data):
        T = len(ep_data['reward'])
        idx = self._get_storage_idx()
        self.states[idx, :T] = ep_data['state']
        self.obs[idx, :T] = ep_data['obs']
        self.actions[idx, :T] = ep_data['action'][:, :, np.newaxis]
        self.rewards[idx, :T] = ep_data['reward'][:, np.newaxis]
        self.dones[idx, :T] = ep_data['done'][:, np.newaxis]
        self.avail_actions[idx, :T] = ep_data['avail_actions']
        self.hidden_states[idx, :T] = ep_data['hidden_state']
        self.filled[idx, :T, :] = True

    def sample(self, batch_size):
        valid_indices = min(self.current_size, self.capacity)
        batch_indices = np.random.choice(valid_indices, batch_size, replace=False)

        # Return tuple MUST be in this exact order
        return (
            torch.tensor(self.states[batch_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.obs[batch_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[batch_indices], dtype=torch.long, device=self.device),
            torch.tensor(self.rewards[batch_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[batch_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.avail_actions[batch_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.hidden_states[batch_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.filled[batch_indices], dtype=torch.float32, device=self.device)
        )

    def _get_storage_idx(self):
        idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)
        return idx

    def __len__(self):
        return self.current_size