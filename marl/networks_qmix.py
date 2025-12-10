# marl/networks_qmix.py
import torch, torch.nn as nn, torch.nn.functional as F, config


class QMIX_RNNAgent(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(QMIX_RNNAgent, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_shape, config.GNN_EMBEDDING_DIM), nn.ReLU(),
                                     nn.Linear(config.GNN_EMBEDDING_DIM, config.GNN_EMBEDDING_DIM), nn.ReLU())
        self.rnn = nn.GRUCell(config.GNN_EMBEDDING_DIM, config.RNN_HIDDEN_DIM)
        self.q_head = nn.Linear(config.RNN_HIDDEN_DIM, action_dim)

    def forward(self, obs_flat, hidden_state):
        emb = self.encoder(obs_flat)
        h_new = self.rnn(emb, hidden_state)
        q = self.q_head(h_new)
        return q, h_new


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super(MixingNetwork, self).__init__()
        self.n_agents, self.state_dim, self.embed_dim = n_agents, state_dim, embed_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs, states):
        b, T = states.shape[:2]
        qs_reshaped = agent_qs.reshape(-1, self.n_agents)
        states_reshaped = states.reshape(-1, self.state_dim)
        qs_for_bmm = qs_reshaped.view(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w1(states_reshaped)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states_reshaped).view(-1, 1, self.embed_dim)
        h = F.elu(torch.bmm(qs_for_bmm, w1) + b1)
        w2 = torch.abs(self.hyper_w2(states_reshaped)).view(-1, self.embed_dim, 1)
        b2 = self.hyper_b2(states_reshaped).view(-1, 1, 1)

        q_total = torch.bmm(h, w2) + b2
        return q_total.view(b, T, -1)