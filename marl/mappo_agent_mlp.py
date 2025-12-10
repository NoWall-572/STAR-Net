# marl/mappo_agent_mlp.py

import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import os
from .networks_baseline import MLPEncoder, Actor, Critic
import config


class MAPPOAgentMLP:
    def __init__(self, num_agents, action_dim_uav, action_dim_ugv, obs_dim, device=None):
        self.num_agents = num_agents
        if device is None:
            self.device = config.DEVICE
        else:
            self.device = torch.device(device)

        self.encoder = MLPEncoder(obs_dim).to(self.device)
        self.actor = Actor(action_dim_uav).to(self.device)
        self.critic = Critic().to(self.device)

        self.params = list(self.encoder.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(self.params, lr=config.LEARNING_RATE, eps=1e-5)

    def choose_action(self, flat_obs, hidden_state, avail_actions, deterministic=False):
        with torch.no_grad():
            flat_obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0).to(self.device)
            avail_tensor = torch.FloatTensor(avail_actions).unsqueeze(0).to(self.device)

            rnn_out = self.encoder(flat_obs_tensor, hidden_state.to(self.device))
            new_hidden_state = rnn_out.cpu()

            dist = self.actor(rnn_out, avail_tensor)
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            action_log_prob = dist.log_prob(action)

            return action.item(), action_log_prob.item(), new_hidden_state

    def update(self, buffer):
        flat_obs_batch, hidden_states, actions, action_log_probs, rewards, dones, _, avail_actions = buffer.get_all()

        flat_obs_batch = torch.FloatTensor(np.array(flat_obs_batch)).to(self.device)
        hidden_states = hidden_states.to(self.device)
        actions = actions.to(self.device)
        action_log_probs = action_log_probs.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        avail_actions = torch.FloatTensor(np.array(avail_actions)).to(self.device)

        T, N, _ = flat_obs_batch.shape

        rnn_outs = []
        for t in range(T):
            rnn_out_t = self.encoder(flat_obs_batch[t], hidden_states[t])
            rnn_outs.append(rnn_out_t)

        rnn_outs = torch.stack(rnn_outs)

        with torch.no_grad():
            values = self.critic(rnn_outs.view(T, N * config.RNN_HIDDEN_DIM))
            next_value = torch.zeros(1, 1).to(self.device)
            advantages = torch.zeros_like(rewards).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]
                delta = rewards[t] + config.GAMMA * next_value * next_non_terminal - values[t]
                advantages[
                    t] = last_gae_lam = delta + config.GAMMA * config.GAE_LAMBDA * next_non_terminal * last_gae_lam
            returns = advantages + values

        explained_var = 1 - torch.var(returns.detach() - values.detach()) / (torch.var(returns.detach()) + 1e-6)

        actions_flat = actions.view(T * N)
        avail_actions_flat = avail_actions.view(T * N, -1)
        old_action_log_probs_flat = action_log_probs.view(T * N)
        advantages_flat = advantages.expand(-1, N).reshape(-1)

        for _ in range(config.PPO_EPOCHS):
            rnn_outs_recomputed = []
            for t in range(T):
                rnn_out_t = self.encoder(flat_obs_batch[t], hidden_states[t])
                rnn_outs_recomputed.append(rnn_out_t)
            rnn_outs_recomputed = torch.stack(rnn_outs_recomputed)
            rnn_outs_flat_recomputed = rnn_outs_recomputed.view(T * N, -1)

            dist = self.actor(rnn_outs_flat_recomputed, avail_actions_flat)
            log_probs_new = dist.log_prob(actions_flat)
            entropy = dist.entropy().mean()

            new_values = self.critic(rnn_outs_recomputed.view(T, N * config.RNN_HIDDEN_DIM))

            # Actor Loss
            ratios = torch.exp(log_probs_new - old_action_log_probs_flat.detach())
            surr1 = ratios * advantages_flat.detach()
            surr2 = torch.clamp(ratios, 1 - config.PPO_CLIP_PARAM, 1 + config.PPO_CLIP_PARAM) * advantages_flat.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic Loss
            critic_loss = nn.MSELoss()(new_values, returns.detach())

            loss = actor_loss + 0.5 * critic_loss - config.ENTROPY_COEFF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()

        diagnostics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'explained_variance': explained_var.item()
        }
        return diagnostics

    def save_models(self, path, episode):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}/actor_{episode}.pth")
        torch.save(self.encoder.state_dict(), f"{path}/encoder_{episode}.pth")

    def load_models(self, actor_path, encoder_path):
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))