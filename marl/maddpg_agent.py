# marl/maddpg_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from .networks import STGATEncoder
from .networks_maddpg import Actor, Critic
import config


class MADDPGAgent:
    def __init__(self, num_agents, action_dim_uav, action_dim_ugv, num_uavs):
        self.device = config.DEVICE
        self.num_agents = num_agents
        self.action_dim_uav = action_dim_uav
        self.action_dim_ugv = action_dim_ugv
        self.action_dim = max(action_dim_uav, action_dim_ugv)
        self.num_uavs = num_uavs

        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.gamma = config.GAMMA
        self.tau = 0.01

        self.encoder = STGATEncoder(config.NODE_FEATURE_DIM).to(self.device)

        self.actors = [Actor(self.action_dim).to(self.device) for _ in range(num_agents)]
        self.critics = [Critic(num_agents, action_dim_uav, action_dim_ugv, num_uavs).to(self.device) for _ in range(num_agents)]

        self.target_actors = [Actor(self.action_dim).to(self.device) for _ in range(num_agents)]
        self.target_critics = [Critic(num_agents, action_dim_uav, action_dim_ugv, num_uavs).to(self.device) for _ in range(num_agents)]

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr_actor)
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=self.lr_actor) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=self.lr_critic) for critic in self.critics]

        for i in range(num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

    def _get_rnn_outs(self, obs_list, hidden_states):
        valid_obs_list = [obs for obs in obs_list if obs['node_features'].size > 0]
        if not valid_obs_list:
            return torch.zeros_like(hidden_states), torch.zeros_like(hidden_states)

        data_list = [Data(x=torch.FloatTensor(obs['node_features']),edge_index=torch.LongTensor(obs['edge_index']),edge_attr=torch.FloatTensor(obs['edge_attr'])) for obs in valid_obs_list]
        batched_data = Batch.from_data_list(data_list).to(self.device)

        node_embeddings = self.encoder.node_encoder(batched_data.x)
        gat_output = self.encoder.gat_conv(node_embeddings, batched_data.edge_index, edge_attr=batched_data.edge_attr)
        agent_indices = batched_data.ptr[:-1]
        agent_embeddings = gat_output[agent_indices]

        new_hidden_states = self.encoder.rnn(agent_embeddings, hidden_states)
        return new_hidden_states, new_hidden_states

    def choose_action(self, observations, hidden_states, avail_actions, explore=True):
        actions = []
        new_hidden_states_list = []
        with torch.no_grad():
            for i in range(self.num_agents):
                obs = observations[i]
                avail = torch.FloatTensor(avail_actions[i]).unsqueeze(0).to(self.device)

                if avail.shape[1] < self.action_dim:
                    padding = torch.zeros((avail.shape[0], self.action_dim - avail.shape[1]), device=self.device)
                    avail = torch.cat([avail, padding], dim=1)

                h_in = hidden_states[i].to(self.device)
                rnn_out, new_hs = self._get_rnn_outs([obs], h_in)
                new_hidden_states_list.append(new_hs.cpu())

                action_probs = self.actors[i](rnn_out, avail)

                if explore:
                    action = torch.distributions.Categorical(action_probs).sample()
                else:
                    action = torch.argmax(action_probs, dim=1)

                actions.append(action.item())

        return actions, new_hidden_states_list

    def update(self, buffer, batch_size):
        if len(buffer) < batch_size:
            return None

        obs_batch, hidden_batch, actions_batch, rewards_batch, dones_batch, \
            next_obs_batch, next_hidden_batch, avail_actions_batch, next_avail_actions_batch = buffer.sample(batch_size)

        hidden_batch = hidden_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        rewards_batch = rewards_batch.to(self.device)
        dones_batch = dones_batch.to(self.device)
        next_hidden_batch = next_hidden_batch.to(self.device)
        avail_actions_batch = avail_actions_batch.to(self.device)
        next_avail_actions_batch = next_avail_actions_batch.to(self.device)

        obs_flat = [obs for obs_list in obs_batch for obs in obs_list]
        hidden_flat = hidden_batch.view(batch_size * self.num_agents, -1)
        rnn_outs_flat, _ = self._get_rnn_outs(obs_flat, hidden_flat)
        rnn_outs = rnn_outs_flat.view(batch_size, self.num_agents, -1)

        next_obs_flat = [obs for obs_list in next_obs_batch for obs in obs_list]
        next_hidden_flat = next_hidden_batch.view(batch_size * self.num_agents, -1)
        next_rnn_outs_flat, _ = self._get_rnn_outs(next_obs_flat, next_hidden_flat)
        next_rnn_outs = next_rnn_outs_flat.view(batch_size, self.num_agents, -1)

        with torch.no_grad():
            target_actions = []
            for i in range(self.num_agents):
                avail = next_avail_actions_batch[:, i, :]
                action_probs = self.target_actors[i](next_rnn_outs[:, i, :], avail)
                target_actions.append(action_probs)

            target_actions_one_hot = torch.stack(target_actions, dim=1)
            rnn_outs_all_cat = next_rnn_outs.view(batch_size, -1)
            target_actions_all_cat = target_actions_one_hot.view(batch_size, -1)
            q_next = self.target_critics[0](rnn_outs_all_cat, target_actions_all_cat)
            q_target = rewards_batch + self.gamma * q_next * (1 - dones_batch)

        actions_one_hot = F.one_hot(actions_batch, num_classes=self.action_dim).float()
        rnn_outs_all_current_cat = rnn_outs.view(batch_size, -1)
        actions_all_current_cat = actions_one_hot.view(batch_size, -1)

        rnn_outs_for_critic = rnn_outs_all_current_cat.detach()

        total_critic_loss = 0
        for i in range(self.num_agents):
            q_current = self.critics[i](rnn_outs_for_critic, actions_all_current_cat)
            critic_loss = F.mse_loss(q_current, q_target.detach())

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            total_critic_loss += critic_loss.item()

        pred_actions = []
        for i in range(self.num_agents):
            avail = avail_actions_batch[:, i, :]
            action_probs = self.actors[i](rnn_outs[:, i, :], avail)
            pred_actions.append(action_probs)
        pred_actions_one_hot = torch.stack(pred_actions, dim=1)
        pred_actions_all_cat = pred_actions_one_hot.view(batch_size, -1)

        self.encoder_optimizer.zero_grad()
        for i in range(self.num_agents):
            self.actor_optimizers[i].zero_grad()

        q_for_actor_update = self.critics[0](rnn_outs_all_current_cat, pred_actions_all_cat)
        actor_loss = -q_for_actor_update.mean()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.encoder_optimizer.step()
        for i in range(self.num_agents):
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optimizers[i].step()

        self.soft_update()

        return {"critic_loss": total_critic_loss / self.num_agents, "actor_loss": actor_loss.item()}

    def soft_update(self):
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self, path, episode):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.encoder.state_dict(), f"{path}/encoder_{episode}.pth")
        for i in range(self.num_agents):
            torch.save(self.actors[i].state_dict(), f"{path}/actor_{i}_{episode}.pth")
            torch.save(self.critics[i].state_dict(), f"{path}/critic_{i}_{episode}.pth")

    def load_models(self, path, episode):
        self.encoder.load_state_dict(torch.load(f"{path}/encoder_{episode}.pth", map_location=self.device))
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(torch.load(f"{path}/actor_{i}_{episode}.pth", map_location=self.device))
            self.critics[i].load_state_dict(torch.load(f"{path}/critic_{i}_{episode}.pth", map_location=self.device))
        print(f"--- Successfully loaded the MADPG model for round {episode} from {path} ---")