# marl/networks_ippo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import config

class DecentralizedCritic(nn.Module):
    def __init__(self):
        super(DecentralizedCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.RNN_HIDDEN_DIM, config.CRITIC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.CRITIC_HIDDEN_DIM, 1)
        )

    def forward(self, rnn_out_one_agent):
        return self.net(rnn_out_one_agent)

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

        if avail_actions.shape[1] > logits.shape[1]:
            avail_actions = avail_actions[:, :logits.shape[1]]
        elif avail_actions.shape[1] < logits.shape[1]:
            padding = torch.zeros((avail_actions.shape[0], logits.shape[1] - avail_actions.shape[1]),
                                  device=logits.device)
            avail_actions = torch.cat([avail_actions, padding], dim=1)

        logits[avail_actions == 0] = -1e10
        dist = Categorical(logits=logits)
        return dist