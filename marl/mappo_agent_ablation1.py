
# marl/mappo_agent_ablation1.py
# Agent for Ablation 1: S-GAT-MAPPO

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
from .networks_ablation1 import SGATEncoder, Actor, Critic
import config
from torch_geometric.data import Data, Batch


class MAPPOAgentAblation1:
    def __init__(self, num_agents, action_dim_uav, action_dim_ugv,device=None):
        self.num_agents = num_agents
        if device is None:
            self.device = config.DEVICE
        else:
            self.device = torch.device(device)
        self.action_dim_uav = action_dim_uav
        self.action_dim_ugv = action_dim_ugv

        self.encoder = SGATEncoder(config.NODE_FEATURE_DIM).to(self.device)
        self.actor = Actor(action_dim_uav).to(self.device)
        self.critic = Critic().to(self.device)

        self.params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(self.params, lr=config.LEARNING_RATE, eps=1e-5)

    def get_initial_hidden_states(self, observations):
        return [torch.zeros(1, config.RNN_HIDDEN_DIM).cpu() for _ in range(len(observations))]

    def choose_action(self, observations, hidden_states, avail_actions, deterministic=False):
        actions, action_log_probs = [], []
        with torch.no_grad():
            for i in range(len(observations)):
                obs = observations[i]
                avail = torch.FloatTensor(avail_actions[i]).unsqueeze(0).to(self.device)
                node_features = torch.FloatTensor(obs['node_features']).to(self.device)
                edge_index = torch.LongTensor(obs['edge_index']).to(self.device)
                edge_attr = torch.FloatTensor(obs['edge_attr']).to(self.device)

                encoder_out = self.encoder(node_features, edge_index, edge_attr, None)

                dist = self.actor(encoder_out, avail)
                action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
                action_log_prob = dist.log_prob(action)

                actions.append(action.item())
                action_log_probs.append(action_log_prob.item())

        new_hidden_states = [torch.zeros(1, config.RNN_HIDDEN_DIM).cpu() for _ in range(len(observations))]
        return actions, action_log_probs, new_hidden_states

    def update(self, buffer):
        obs, _, actions, action_log_probs, rewards, dones, states, avail_actions = buffer.get_all()

        actions = actions.to(self.device)
        action_log_probs = action_log_probs.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            all_encoder_outs = self._recompute_all_encoder_outs(obs)
            values = self.critic(all_encoder_outs)
            advantages = torch.zeros_like(rewards).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = torch.zeros(1, 1).to(self.device)
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]
                delta = rewards[t] + config.GAMMA * next_value * next_non_terminal - values[t]
                advantages[
                    t] = last_gae_lam = delta + config.GAMMA * config.GAE_LAMBDA * next_non_terminal * last_gae_lam
            returns = advantages + values
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        explained_var = 1 - torch.var(returns.detach() - values.detach()) / (torch.var(returns.detach()) + 1e-6)

        for _ in range(config.PPO_EPOCHS):
            log_probs_new, state_values_new, dist_entropy_new = self.evaluate_actions(obs, actions, avail_actions)

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

        diagnostics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': dist_entropy_new.item(),
            'explained_variance': explained_var.item()
        }
        return diagnostics

    def _recompute_all_encoder_outs(self, obs_batch):
        T = len(obs_batch)
        N = self.num_agents
        all_obs_flat = [obs for obs_t in obs_batch for obs in obs_t]

        data_list = [Data(x=torch.FloatTensor(obs['node_features']),
                          edge_index=torch.LongTensor(obs['edge_index']),
                          edge_attr=torch.FloatTensor(obs['edge_attr'])) for obs in all_obs_flat]
        batched_data = Batch.from_data_list(data_list).to(self.device)

        encoded_nodes = F.relu(self.encoder.node_encoder(batched_data.x))
        gat_output = self.encoder.gat_conv(encoded_nodes, batched_data.edge_index, edge_attr=batched_data.edge_attr)
        agent_indices = batched_data.ptr[:-1]
        agent_embeddings = gat_output[agent_indices]

        return agent_embeddings.view(T, N * config.RNN_HIDDEN_DIM)

    def evaluate_actions(self, obs_batch, action_batch, avail_batch):
        T = len(obs_batch)
        N = self.num_agents
        all_obs_flat = [obs for obs_t in obs_batch for obs in obs_t]

        data_list = [Data(x=torch.FloatTensor(obs['node_features']),
                          edge_index=torch.LongTensor(obs['edge_index']),
                          edge_attr=torch.FloatTensor(obs['edge_attr'])) for obs in all_obs_flat]
        batched_data = Batch.from_data_list(data_list).to(self.device)

        encoded_nodes = F.relu(self.encoder.node_encoder(batched_data.x))
        gat_output = self.encoder.gat_conv(encoded_nodes, batched_data.edge_index, edge_attr=batched_data.edge_attr)
        agent_indices = batched_data.ptr[:-1]
        encoder_outs_flat = gat_output[agent_indices]

        avail_actions_flat = torch.FloatTensor(np.array(avail_batch)).view(T * N, -1).to(self.device)
        dist = self.actor(encoder_outs_flat, avail_actions_flat)
        log_probs_flat = dist.log_prob(action_batch.view(-1))
        log_probs_new = log_probs_flat.view(T, N)
        entropy = dist.entropy().mean()

        encoder_outs_for_critic = encoder_outs_flat.view(T, N * config.RNN_HIDDEN_DIM)
        state_values = self.critic(encoder_outs_for_critic)

        return log_probs_new, state_values, entropy

    def save_models(self, path, episode):
        if not os.path.exists(path): os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}/actor_{episode}.pth")
        torch.save(self.encoder.state_dict(), f"{path}/encoder_{episode}.pth")

    def load_models(self, actor_path, encoder_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))