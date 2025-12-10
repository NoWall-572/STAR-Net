# marl/qmix_agent.py
import torch, os
from copy import deepcopy
from .networks_qmix import QMIX_RNNAgent, MixingNetwork
import config


class QMIXAgent:
    def __init__(self, n_agents, action_dim, state_dim, obs_dim, device=None):
        self.num_agents, self.device = n_agents, device or torch.device(config.DEVICE)
        self.gamma, self.lr, self.clip, self.tau = config.GAMMA, config.LEARNING_RATE, 1.0, 0.005

        self.agent_net = QMIX_RNNAgent(obs_dim, action_dim).to(self.device)
        self.mixer = MixingNetwork(n_agents, state_dim).to(self.device)
        self.target_agent_net = deepcopy(self.agent_net)
        self.target_mixer = deepcopy(self.mixer)

        self.params = list(self.agent_net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr, eps=1e-5)

    def train(self, batch):
        states, obs, actions, rewards, dones, avail, hidden, filled = batch
        b, T, n, _ = obs.shape

        # --- Target Q ---
        with torch.no_grad():
            h_target = hidden[:, 0].reshape(b * n, -1)
            q_targets_steps = []
            for t in range(T):
                q_t, h_target = self.target_agent_net(obs[:, t].reshape(-1, obs.shape[-1]), h_target)
                q_targets_steps.append(q_t.reshape(b, n, -1))
            q_targets = torch.stack(q_targets_steps, dim=1)
            q_targets[avail == 0] = -999999
            max_q_targets = q_targets.max(dim=-1)[0]
            q_total_target = self.target_mixer(max_q_targets, states)
            target = rewards + self.gamma * (1 - dones) * q_total_target

        # --- Eval Q ---
        h_eval = hidden[:, 0].reshape(b * n, -1)
        q_evals_steps = []
        for t in range(T):
            q_t, h_eval = self.agent_net(obs[:, t].reshape(-1, obs.shape[-1]), h_eval)
            q_evals_steps.append(q_t.reshape(b, n, -1))
        q_evals = torch.stack(q_evals_steps, dim=1)
        chosen_q_vals = torch.gather(q_evals, dim=-1, index=actions).squeeze(-1)
        q_total_current = self.mixer(chosen_q_vals, states)

        # --- Loss ---
        loss = ((q_total_current - target.detach()) * filled) ** 2
        loss = loss.sum() / filled.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        return loss.item()

    def _update_targets(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def update_targets(self):
        self._update_targets(self.target_agent_net, self.agent_net)
        self._update_targets(self.target_mixer, self.mixer)

    def save(self, path, ep):
        if not os.path.exists(path): os.makedirs(path)
        torch.save(self.agent_net.state_dict(), f"{path}/agent_{ep}.pth")
        torch.save(self.mixer.state_dict(), f"{path}/mixer_{ep}.pth")
        print(f"--- Saved QMIX model at episode {ep} ---")

    def load(self, path, ep):
        self.agent_net.load_state_dict(torch.load(f"{path}/agent_{ep}.pth", map_location=self.device))
        self.mixer.load_state_dict(torch.load(f"{path}/mixer_{ep}.pth", map_location=self.device))
        self.update_targets()
        print(f"--- Loaded QMIX model from episode {ep} ---")

    def choose_action(self, flat_obs_list, hidden_states_list, avail_actions_list, deterministic=True):
        actions = []
        new_hidden_states = []

        with torch.no_grad():
            for i in range(self.num_agents):
                obs_i_flat = torch.FloatTensor(flat_obs_list[i]).unsqueeze(0).to(self.device)

                hidden_state_i = hidden_states_list[i].to(self.device)

                q_vals, new_h = self.agent_net(obs_i_flat, hidden_state_i)
                new_hidden_states.append(new_h.squeeze(0))

                avail_tensor = torch.FloatTensor(avail_actions_list[i]).unsqueeze(0).to(self.device)
                q_vals[avail_tensor == 0] = -float('inf')
                action = q_vals.argmax().item()
                actions.append(action)

        return actions, new_hidden_states