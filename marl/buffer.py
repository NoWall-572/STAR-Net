# marl/buffer.py

import torch
import numpy as np

class OnPolicyBuffer:
    def __init__(self):
        self.obs = []
        self.hidden_states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.avail_actions = []

    def push(self, obs, hidden_states, action, action_log_prob, reward, done, state, avail_actions):
        self.obs.append(obs)
        self.hidden_states.append([h.cpu().numpy().squeeze(0) for h in hidden_states])
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.states.append(state)
        self.avail_actions.append(avail_actions)

    def get_all(self):
        return (
            self.obs,
            torch.FloatTensor(np.array(self.hidden_states)),
            torch.LongTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.action_log_probs)),
            torch.FloatTensor(np.array(self.rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(self.dones)).unsqueeze(1),
            self.states,
            self.avail_actions
        )

    def clear(self):
        self.__init__()