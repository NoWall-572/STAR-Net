# marl/networks_ablation1.py
# Ablation 1: S-GAT-MAPPO (Removed Temporal Modeling via GRU)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.distributions import Categorical
import config


class SGATEncoder(nn.Module):
    def __init__(self, node_feature_dim):
        super(SGATEncoder, self).__init__()
        self.node_encoder = nn.Linear(node_feature_dim, config.GNN_EMBEDDING_DIM)
        self.gat_conv = GATv2Conv(
            in_channels=config.GNN_EMBEDDING_DIM,
            out_channels=config.GNN_EMBEDDING_DIM,
            heads=4,
            concat=False,
            dropout=0.1,
            edge_dim=9
        )

    def forward(self, node_features, edge_index, edge_attr, hidden_state):
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(0)
        encoded_nodes = F.relu(self.node_encoder(node_features))
        gat_output = self.gat_conv(encoded_nodes, edge_index, edge_attr=edge_attr)
        agent_embedding = gat_output[0, :].unsqueeze(0)

        if config.GNN_EMBEDDING_DIM != config.RNN_HIDDEN_DIM:
            pass

        return agent_embedding

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
        if avail_actions.shape[1] < logits.shape[1]:
            padding = torch.zeros((1, logits.shape[1] - avail_actions.shape[1]), device=logits.device)
            avail_actions = torch.cat([avail_actions, padding], dim=1)

        logits[avail_actions == 0] = -1e10
        dist = Categorical(logits=logits)
        return dist


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        critic_input_dim = (config.NUM_UAVS + config.NUM_UGVS) * config.RNN_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(critic_input_dim, config.CRITIC_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.CRITIC_HIDDEN_DIM, 1)
        )

    def forward(self, rnn_outs_all_agents):
        state = rnn_outs_all_agents.reshape(rnn_outs_all_agents.size(0), -1)
        return self.net(state)

