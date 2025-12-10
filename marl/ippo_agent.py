# marl/ippo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from .networks import STGATEncoder
from .networks_ippo import Actor, DecentralizedCritic
import config
from torch_geometric.data import Data, Batch


class IPPOAgent:
    def __init__(self, num_agents, action_dim_uav, action_dim_ugv, device=None):
        self.num_agents = num_agents
        if device is None:
            self.device = config.DEVICE
        else:
            self.device = torch.device(device)
        self.action_dim_uav = action_dim_uav
        self.action_dim_ugv = action_dim_ugv

        self.encoder = STGATEncoder(config.NODE_FEATURE_DIM).to(self.device)
        self.actor = Actor(action_dim_uav).to(self.device)
        self.critic = DecentralizedCritic().to(self.device)

        self.params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(self.params, lr=config.LEARNING_RATE, eps=1e-5)

    def get_initial_hidden_states(self, observations):
        initial_hidden_states = []
        with torch.no_grad():
            for i in range(len(observations)):
                obs = observations[i]
                node_features = torch.FloatTensor(obs['node_features']).to(self.device)
                edge_index = torch.LongTensor(obs['edge_index']).to(self.device)
                edge_attr = torch.FloatTensor(obs['edge_attr']).to(self.device)
                initial_h = self.encoder(node_features, edge_index, edge_attr, None)
                initial_hidden_states.append(initial_h.cpu())
        return initial_hidden_states

    def choose_action(self, observations, hidden_states, avail_actions, deterministic=False):
        actions, action_log_probs, new_hidden_states = [], [], []
        with torch.no_grad():
            for i in range(len(observations)):
                obs = observations[i]
                avail = torch.FloatTensor(avail_actions[i]).unsqueeze(0).to(self.device)
                node_features = torch.FloatTensor(obs['node_features']).to(self.device)
                edge_index = torch.LongTensor(obs['edge_index']).to(self.device)
                edge_attr = torch.FloatTensor(obs['edge_attr']).to(self.device)
                rnn_out = self.encoder(node_features, edge_index, edge_attr, hidden_states[i].to(self.device))
                new_hidden_states.append(rnn_out.cpu())
                dist = self.actor(rnn_out, avail)
                action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
                action_log_prob = dist.log_prob(action)
                actions.append(action.item())
                action_log_probs.append(action_log_prob.item())
        return actions, action_log_probs, new_hidden_states

    def update(self, buffer):
        obs, hidden_states, actions, action_log_probs, rewards, dones, states, avail_actions = buffer.get_all()

        hidden_states = hidden_states.to(self.device)
        actions = actions.to(self.device)
        action_log_probs = action_log_probs.to(self.device)

        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            all_rnn_outs_flat = self._recompute_all_rnn_outs_flat(obs, hidden_states)
            values_flat = self.critic(all_rnn_outs_flat)
            values = values_flat.view(len(obs), self.num_agents)

            advantages = torch.zeros_like(values).to(self.device)
            last_gae_lam = torch.zeros(self.num_agents).to(self.device)

            for t in reversed(range(rewards.shape[0])):
                if t == rewards.shape[0] - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = torch.zeros(self.num_agents).to(self.device)
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]

                delta = rewards[t] + config.GAMMA * next_value * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + config.GAMMA * config.GAE_LAMBDA * next_non_terminal * last_gae_lam

            returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        explained_var = 1 - torch.var(returns.detach() - values.detach()) / (torch.var(returns.detach()) + 1e-6)

        for _ in range(config.PPO_EPOCHS):
            log_probs_new, state_values_new, dist_entropy_new = self.evaluate_actions(obs, hidden_states, actions,avail_actions)

            ratios = torch.exp(log_probs_new - action_log_probs.detach())
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - config.PPO_CLIP_PARAM, 1 + config.PPO_CLIP_PARAM) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(state_values_new, returns.detach())

            loss = actor_loss + 0.5 * critic_loss - config.ENTROPY_COEFF * dist_entropy_new

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': dist_entropy_new.item(),
            'explained_variance': explained_var.item()
        }

    def _recompute_all_rnn_outs_flat(self, obs_batch, hidden_batch):
        T = len(obs_batch);
        N = self.num_agents
        all_obs_flat = [obs for obs_t in obs_batch for obs in obs_t]
        data_list = [Data(x=torch.FloatTensor(o['node_features']), edge_index=torch.LongTensor(o['edge_index']),edge_attr=torch.FloatTensor(o['edge_attr'])) for o in all_obs_flat]
        batched_data = Batch.from_data_list(data_list).to(self.device)
        hidden_states_flat = hidden_batch.view(T * N, -1)
        node_embeddings = self.encoder.node_encoder(batched_data.x)
        gat_output = self.encoder.gat_conv(node_embeddings, batched_data.edge_index, edge_attr=batched_data.edge_attr)
        agent_indices = batched_data.ptr[:-1]
        agent_embeddings = gat_output[agent_indices]
        return self.encoder.rnn(agent_embeddings, hidden_states_flat)

    def evaluate_actions(self, obs_batch, hidden_batch, action_batch, avail_batch):
        T = len(obs_batch);
        N = self.num_agents
        rnn_outs_flat = self._recompute_all_rnn_outs_flat(obs_batch, hidden_batch)
        avail_actions_flat = torch.FloatTensor(np.array(avail_batch)).view(T * N, -1).to(self.device)
        dist = self.actor(rnn_outs_flat, avail_actions_flat)
        log_probs_flat = dist.log_prob(action_batch.view(-1))
        log_probs_new = log_probs_flat.view(T, N)
        entropy = dist.entropy().mean()
        state_values_flat = self.critic(rnn_outs_flat)
        state_values_new = state_values_flat.view(T, N)
        return log_probs_new, state_values_new, entropy

    def save_models(self, path, episode):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}/actor_ippo_{episode}.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic_ippo_{episode}.pth")
        torch.save(self.encoder.state_dict(), f"{path}/encoder_ippo_{episode}.pth")

    def load_models(self, actor_path, critic_path, encoder_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))