# marl/networks_maddpg.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.RNN_HIDDEN_DIM, config.ACTOR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.ACTOR_HIDDEN_DIM, action_dim)
        )

    def forward(self, rnn_out, avail_actions):
        logits = self.net(rnn_out)
        logits[avail_actions == 0] = -1e10
        action_probs = F.gumbel_softmax(logits, hard=True, dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, num_agents, action_dim_uav, action_dim_ugv, num_uavs):
        super(Critic, self).__init__()

        self.action_dim = max(action_dim_uav, action_dim_ugv)

        critic_input_dim = num_agents * config.RNN_HIDDEN_DIM + num_agents * self.action_dim

        self.net = nn.Sequential(
            nn.Linear(critic_input_dim, config.CRITIC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.CRITIC_HIDDEN_DIM, config.CRITIC_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.CRITIC_HIDDEN_DIM // 2, 1)
        )

    def forward(self, rnn_outs_all, actions_all_one_hot):
        # rnn_outs_all shape: [batch_size, num_agents * rnn_hidden_dim]
        # actions_all_one_hot shape: [batch_size, num_agents * action_dim]
        state_action_input = torch.cat([rnn_outs_all, actions_all_one_hot], dim=1)
        return self.net(state_action_input)