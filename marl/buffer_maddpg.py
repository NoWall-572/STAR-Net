# marl/buffer_maddpg.py

import torch
import numpy as np
import config


class ReplayBuffer:
    def __init__(self, capacity, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer_counter = 0

        self.obs = [[] for _ in range(num_agents)]
        self.hidden_states = [[] for _ in range(num_agents)]
        self.actions = [[] for _ in range(num_agents)]
        self.rewards = []
        self.dones = []
        self.next_obs = [[] for _ in range(num_agents)]
        self.next_hidden_states = [[] for _ in range(num_agents)]
        self.avail_actions = [[] for _ in range(num_agents)]
        self.next_avail_actions = [[] for _ in range(num_agents)]

    def push(self, obs, hidden_states, actions, reward, next_obs, next_hidden_states, done, avail_actions,
             next_avail_actions):
        idx = self.buffer_counter % self.capacity

        if self.buffer_counter < self.capacity:
            for i in range(self.num_agents):
                self.obs[i].append(obs[i])
                self.hidden_states[i].append(hidden_states[i].cpu().numpy())
                self.actions[i].append(actions[i])
                self.next_obs[i].append(next_obs[i])
                self.next_hidden_states[i].append(next_hidden_states[i].cpu().numpy())
                self.avail_actions[i].append(avail_actions[i])
                self.next_avail_actions[i].append(next_avail_actions[i])
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            for i in range(self.num_agents):
                self.obs[i][idx] = obs[i]
                self.hidden_states[i][idx] = hidden_states[i].cpu().numpy()
                self.actions[i][idx] = actions[i]
                self.next_obs[i][idx] = next_obs[i]
                self.next_hidden_states[i][idx] = next_hidden_states[i].cpu().numpy()
                self.avail_actions[i][idx] = avail_actions[i]
                self.next_avail_actions[i][idx] = next_avail_actions[i]
            self.rewards[idx] = reward
            self.dones[idx] = done

        self.buffer_counter += 1

    def sample(self, batch_size):
        num_samples = min(self.buffer_counter, self.capacity)
        indices = np.random.choice(num_samples, batch_size, replace=False)

        obs_batch = [[self.obs[i][j] for j in indices] for i in range(self.num_agents)]
        hidden_batch = torch.FloatTensor(
            np.array([self.hidden_states[i][j] for j in indices for i in range(self.num_agents)]).reshape(batch_size,self.num_agents,-1))
        actions_batch = torch.LongTensor(
            np.array([self.actions[i][j] for j in indices for i in range(self.num_agents)]).reshape(batch_size,self.num_agents))
        rewards_batch = torch.FloatTensor(np.array(self.rewards)[indices]).unsqueeze(1)
        dones_batch = torch.FloatTensor(np.array(self.dones)[indices]).unsqueeze(1)
        next_obs_batch = [[self.next_obs[i][j] for j in indices] for i in range(self.num_agents)]
        next_hidden_batch = torch.FloatTensor(
            np.array([self.next_hidden_states[i][j] for j in indices for i in range(self.num_agents)]).reshape(batch_size, self.num_agents, -1))
        avail_actions_batch = torch.FloatTensor(
            np.array([self.avail_actions[i][j] for j in indices for i in range(self.num_agents)]).reshape(batch_size,self.num_agents,-1))
        next_avail_actions_batch = torch.FloatTensor(
            np.array([self.next_avail_actions[i][j] for j in indices for i in range(self.num_agents)]).reshape(batch_size, self.num_agents, -1))

        return obs_batch, hidden_batch, actions_batch, rewards_batch, dones_batch, next_obs_batch, next_hidden_batch, avail_actions_batch, next_avail_actions_batch

    def __len__(self):
        return min(self.buffer_counter, self.capacity)