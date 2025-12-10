# environment/env.py (Optimized Version)

import numpy as np
import networkx as nx
import config
from .entities import UAV, UGV
from . import channel_models
from .threats import EnvironmentalThreats, Jammer
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rice

class AirGroundEnv:
    def __init__(self, num_uavs=None, num_ugvs=None):
        self.num_uavs = num_uavs if num_uavs is not None else config.NUM_UAVS
        self.num_ugvs = num_ugvs if num_ugvs is not None else config.NUM_UGVS
        self.num_agents = self.num_uavs + self.num_ugvs
        self.area_size = config.AREA_SIZE
        self.sinr_threshold_linear = 10 ** (config.SINR_THRESHOLD_dB / 10)
        self.env_threats = EnvironmentalThreats()
        self.jammer = Jammer('jammer_0', config.JAMMER_POSITION, config.JAMMER_POWER_dBm, config.JAMMER_GAIN_dBi)

        self.nodes, self.uavs, self.ugvs = [], [], []
        self.positions = np.zeros((self.num_agents, 3))
        self.velocities = np.zeros((self.num_agents, 3))
        self.remaining_energies = np.zeros(self.num_uavs)
        self.max_speeds = np.zeros(self.num_agents)
        self.max_accelerations = np.zeros(self.num_agents)
        self.is_uav_mask = np.zeros(self.num_agents, dtype=bool)
        self.uav_indices = np.arange(self.num_uavs)

        self.transmit_powers = np.zeros(self.num_agents)

        self.G = nx.Graph()
        self.last_step_total_energy = 0.0
        self.last_sinr_matrix = np.zeros((self.num_agents, self.num_agents))
        self.last_adjacency_matrix = np.zeros((self.num_agents, self.num_agents), dtype=bool)
        self.last_gain_matrix = np.zeros((self.num_agents, self.num_agents))
        self.last_rate_matrix = np.zeros((self.num_agents, self.num_agents))
        self.num_agents = self.num_uavs + self.num_ugvs
        self.area_size = config.AREA_SIZE

        self.node_id_to_idx = {}

    def reset(self,scenario_config=None):
        self._create_agents(scenario_config=scenario_config)
        self.node_id_to_idx = {node.id: i for i, node in enumerate(self.nodes)}
        initial_power_watts = channel_models.dbm_to_watts(config.NODE_TRANSMIT_POWER_dBm / 2)
        self.transmit_powers = np.full(self.num_agents, initial_power_watts)
        for i, node in enumerate(self.nodes):
            node.reset()
            self.positions[i] = node.pos
            self.velocities[i] = node.vel
            self.max_speeds[i] = node.max_speed
            self.max_accelerations[i] = node.max_acceleration
            if isinstance(node, UAV):
                self.is_uav_mask[i] = True
                uav_idx = self.uavs.index(node)
                self.remaining_energies[uav_idx] = node.remaining_energy
            else:
                self.is_uav_mask[i] = False

        self.last_step_total_energy = self.num_uavs * config.UAV_INITIAL_ENERGY

        initial_powers = np.full(self.num_agents, channel_models.dbm_to_watts(config.NODE_TRANSMIT_POWER_dBm))
        if config.SCENARIO_TYPE == 'ADVERSARIAL' and self.jammer is not None:
            self._update_network_graph(initial_powers)

            if self.G.number_of_edges() > 0:
                try:
                    edge_centrality = nx.edge_betweenness_centrality(self.G)
                except Exception as e:
                    print(f"Warning: Could not compute edge centrality, graph might be disconnected. Error: {e}")
                    edge_centrality = {}

                if edge_centrality:
                    target_edge = max(edge_centrality, key=edge_centrality.get)
                    pos1 = self.positions[target_edge[0]]
                    pos2 = self.positions[target_edge[1]]
                    jammer_pos = (pos1 + pos2) / 2.0

                    self.jammer.set_position(jammer_pos)

        self._update_network_graph(initial_powers)

        self._sync_objects_from_state_arrays()

        return self._get_obs(), self._get_state()

    def _sync_objects_from_state_arrays(self):
        for i, node in enumerate(self.nodes):
            node.pos = self.positions[i]
            node.vel = self.velocities[i]
            if isinstance(node, UAV):
                try:
                    uav_idx = self.uavs.index(node)
                    node.remaining_energy = self.remaining_energies[uav_idx]
                except ValueError:
                    pass

    def step(self, actions_with_powers):
        num_nodes = len(self.nodes)
        accel_cmds = np.zeros((num_nodes, 3))
        transmit_powers_watts = np.zeros(num_nodes)
        for node_id, action in actions_with_powers.items():
            if node_id in self.node_id_to_idx:
                idx = self.node_id_to_idx[node_id]
                accel_cmds[idx] = action['acceleration']
                transmit_powers_watts[idx] = action['transmit_power_watts']

        self.transmit_powers = transmit_powers_watts

        accel_norm = np.linalg.norm(accel_cmds, axis=1)
        scale = np.minimum(1.0, self.max_accelerations / (accel_norm + 1e-9))
        clipped_accel = accel_cmds * scale[:, np.newaxis]

        self.velocities += clipped_accel * config.SIM_TIME_STEP

        speed_norm = np.linalg.norm(self.velocities, axis=1)
        scale = np.minimum(1.0, self.max_speeds / (speed_norm + 1e-9))
        self.velocities *= scale[:, np.newaxis]

        self.velocities[~self.is_uav_mask, 2] = 0.0

        self.positions += self.velocities * config.SIM_TIME_STEP
        self.positions[~self.is_uav_mask, 2] = 0.0

        wind_vec = self.env_threats.get_wind_vector()
        uav_velocities = self.velocities[self.is_uav_mask]
        uav_tx_powers = transmit_powers_watts[self.is_uav_mask]

        p_base = config.UAV_P_BASE_COEFF * (np.linalg.norm(uav_velocities, axis=1) ** 3)
        relative_wind_speed = np.linalg.norm(uav_velocities - wind_vec, axis=1)
        p_wind = config.UAV_P_WIND_COEFF * (relative_wind_speed ** 2)
        p_prop = p_base + p_wind
        p_comm = config.UAV_P_COMM_COEFF * uav_tx_powers
        total_power = p_prop + p_comm
        energy_consumed = total_power * config.SIM_TIME_STEP

        self.remaining_energies -= energy_consumed
        self.remaining_energies = np.maximum(0, self.remaining_energies)

        self._update_network_graph(transmit_powers_watts)
        reward, reward_components = self._calculate_reward()

        done = np.any(self.remaining_energies <= 0)

        self._sync_objects_from_state_arrays()

        return self._get_obs(), self._get_state(), reward, done, reward_components

    def _create_agents(self, scenario_config=None):
        uavs, ugvs, nodes = [], [], []

        if scenario_config is None:
            for i in range(self.num_uavs):
                pos = [np.random.uniform(0, self.area_size), np.random.uniform(0, self.area_size),
                       np.random.uniform(50, 150)]
                uav = UAV(node_id=f'uav_{i}', initial_position=pos)
                uavs.append(uav)
                nodes.append(uav)
            for i in range(self.num_ugvs):
                pos = [np.random.uniform(0, self.area_size), np.random.uniform(0, self.area_size)]
                ugv = UGV(node_id=f'ugv_{i}', initial_position=pos)
                ugvs.append(ugv)
                nodes.append(ugv)
        else:
            print("Loading the initial position of the AI agent from the preset scene...")
            for i, pos in enumerate(scenario_config['uav_positions']):
                uav = UAV(node_id=f'uav_{i}', initial_position=pos)
                uavs.append(uav)
                nodes.append(uav)
            for i, pos in enumerate(scenario_config['ugv_positions']):
                ugv = UGV(node_id=f'ugv_{i}', initial_position=pos)
                ugvs.append(ugv)
                nodes.append(ugv)

        self.uavs, self.ugvs, self.nodes = uavs, ugvs, nodes

    def _update_network_graph(self, transmit_powers_watts):
        distance_matrix = squareform(pdist(self.positions))

        gain_matrix = self._get_vectorized_channel_gains(self.positions, distance_matrix)

        received_power_matrix = transmit_powers_watts[:, np.newaxis] * gain_matrix
        jammer_interferences = self.jammer.get_jamming_interference(self.positions)
        noise_watts = channel_models.dbm_to_watts(config.NOISE_POWER_dBm)

        sinr_matrix = received_power_matrix / (noise_watts + jammer_interferences[np.newaxis, :])
        np.nan_to_num(sinr_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        adjacency_matrix = (sinr_matrix >= self.sinr_threshold_linear) & (sinr_matrix.T >= self.sinr_threshold_linear)
        np.fill_diagonal(adjacency_matrix, False)

        self.G = nx.from_numpy_array(adjacency_matrix)

        degrees = np.sum(adjacency_matrix, axis=1)
        degrees[degrees == 0] = 1
        bandwidth_per_link = config.BANDWIDTH / degrees[:, np.newaxis]
        rate_matrix = bandwidth_per_link * np.log2(1 + sinr_matrix)
        np.nan_to_num(rate_matrix, copy=False, nan=0.0)
        # +++++++++++++++++++++++++++++++++++

        self.last_sinr_matrix = sinr_matrix
        self.last_adjacency_matrix = np.triu(adjacency_matrix)
        self.last_gain_matrix = gain_matrix
        self.last_rate_matrix = rate_matrix

    def _get_vectorized_channel_gains(self, all_pos, distance_matrix):
        N = self.num_agents
        gain_matrix = np.zeros((N, N))
        is_ugv_mask = ~self.is_uav_mask

        a2a_mask = self.is_uav_mask[:, np.newaxis] & self.is_uav_mask
        a2g_mask = (self.is_uav_mask[:, np.newaxis] & is_ugv_mask) | (is_ugv_mask[:, np.newaxis] & self.is_uav_mask)
        g2g_mask = is_ugv_mask[:, np.newaxis] & is_ugv_mask

        # A2A
        if np.any(a2a_mask):
            loss_db_a2a = channel_models.get_A2A_path_loss(distance_matrix[a2a_mask])
            gain_matrix[a2a_mask] = channel_models.path_loss_to_gain(loss_db_a2a)
        # G2G
        if np.any(g2g_mask):
            dist_g2g = distance_matrix[g2g_mask]
            dist_g2g[dist_g2g <= 0] = 1e-6

            avg_loss_db_g2g = 10 * config.G2G_PATH_LOSS_EXPONENT_N * np.log10(
                dist_g2g) + config.G2G_PATH_LOSS_CONSTANT_C
            avg_gain_g2g = channel_models.path_loss_to_gain(avg_loss_db_g2g)

            K = config.G2G_RICIAN_K_FACTOR

            h_los_gain = np.sqrt(K / (K + 1))

            num_g2g_links = np.sum(g2g_mask)
            real_part = np.random.randn(num_g2g_links) / np.sqrt(2)
            imag_part = np.random.randn(num_g2g_links) / np.sqrt(2)
            s = np.sqrt(2 * K)
            scale = 1 / np.sqrt(2 * (K + 1))
            fading_power_factor = rice.rvs(b=s, scale=scale, size=num_g2g_links) ** 2

            instantaneous_gain_g2g = avg_gain_g2g * fading_power_factor

            gain_matrix[g2g_mask] = instantaneous_gain_g2g
        # A2G
        if np.any(a2g_mask):
            uav_indices, ugv_indices = np.where(self.is_uav_mask[:, np.newaxis] & is_ugv_mask)
            loss_db_a2g = channel_models.get_A2G_path_loss(all_pos[uav_indices], all_pos[ugv_indices])
            gains_a2g = channel_models.path_loss_to_gain(loss_db_a2g)
            gain_matrix[uav_indices, ugv_indices] = gains_a2g
            gain_matrix[ugv_indices, uav_indices] = gains_a2g

        return gain_matrix

    def _get_channel_gain(self, sender, receiver):
        pass

    def _get_all_node_features(self):
        features = np.zeros((self.num_agents, config.NODE_FEATURE_DIM))

        features[:, 0:3] = self.positions / self.area_size
        features[:, 3:6] = self.velocities / (self.max_speeds[:, np.newaxis] + 1e-6)

        uav_energies_norm = self.remaining_energies / config.UAV_INITIAL_ENERGY
        features[self.is_uav_mask, 6] = uav_energies_norm
        features[~self.is_uav_mask, 6] = 1.0

        max_power_watts = channel_models.dbm_to_watts(config.NODE_TRANSMIT_POWER_dBm)
        normalized_powers = self.transmit_powers / (max_power_watts + 1e-9)
        features[:, 7] = normalized_powers

        if self.G.number_of_nodes() > 0:
            centrality = nx.degree_centrality(self.G)
            features[:, 8] = [centrality.get(i, 0) for i in range(self.num_agents)]

        jammer_interferences = self.jammer.get_jamming_interference(self.positions)
        features[:, 9] = np.tanh(jammer_interferences / 1e-9)
        features[:, 10] = config.RAIN_RATE

        features[self.is_uav_mask, 11] = 1.0
        features[~self.is_uav_mask, 12] = 1.0

        return features

    def _get_state(self):
        return self._get_all_node_features().flatten()

    def _get_obs(self):
        return [self._get_one_agent_obs(i) for i in range(self.num_agents)]

    def _get_one_agent_obs(self, agent_id):
        all_node_features = self._get_all_node_features()
        ego_graph = nx.ego_graph(self.G, agent_id, radius=1)
        nodes_in_subgraph = list(ego_graph.nodes())
        if agent_id not in nodes_in_subgraph:
            nodes_in_subgraph.insert(0, agent_id)
        else:
            nodes_in_subgraph.remove(agent_id)
            nodes_in_subgraph.insert(0, agent_id)
        mapping = {old_id: new_id for new_id, old_id in enumerate(nodes_in_subgraph)}
        subgraph_node_features = all_node_features[nodes_in_subgraph]
        edge_list = list(ego_graph.edges())
        if not edge_list:
            edge_index = np.array([[], []], dtype=int)
        else:
            remapped_edges = np.array([[mapping[u], mapping[v]] for u, v in edge_list]).T
            edge_index = np.concatenate([remapped_edges, remapped_edges[[1, 0], :]], axis=1)
        edge_attr_list = []
        for u, v in edge_list:
            dist = np.linalg.norm(self.positions[u] - self.positions[v])
            rel_vel = np.linalg.norm(self.velocities[u] - self.velocities[v])
            sinr_uv = self.last_sinr_matrix[u, v]
            sinr_vu = self.last_sinr_matrix[v, u]
            gain_uv = self.last_gain_matrix[u, v]
            rate_uv = self.last_rate_matrix[u, v]

            is_u_uav = self.is_uav_mask[u]
            is_v_uav = self.is_uav_mask[v]
            link_type = 0.0
            if not is_u_uav and not is_v_uav:
                link_type = 2.0
            elif is_u_uav != is_v_uav:
                link_type = 1.0
            link_type_one_hot = [0.0, 0.0, 0.0]
            link_type_one_hot[int(link_type)] = 1.0

            features = [
                dist / self.area_size,
                rel_vel / config.UAV_MAX_SPEED,
                np.tanh(sinr_uv),
                np.tanh(sinr_vu),
                np.log10(gain_uv + 1e-12),
                rate_uv / config.BANDWIDTH,
            ]
            features.extend(link_type_one_hot)
            edge_attr_list.append(features)

        if not edge_attr_list:
            edge_attr = np.empty((0, 9), dtype=np.float32)
        else:
            edge_attr = np.concatenate([edge_attr_list, edge_attr_list], axis=0)
        # ++++++++++++++++++++++++++++

        return {'node_features': subgraph_node_features, 'edge_index': edge_index, 'edge_attr': edge_attr}

    def get_avail_actions(self):
        avail_actions = []
        num_power_levels = config.ACTION_POWER_LEVELS
        uav_energy_threshold = config.UAV_INITIAL_ENERGY * 0.05

        for i, node in enumerate(self.nodes):
            if isinstance(node, UAV):
                uav_idx = self.uavs.index(node)
                current_energy = self.remaining_energies[uav_idx]

                num_move_actions = config.ACTION_MOVE_LEVELS_UAV
                avail_agent_actions = [1] * (num_move_actions * num_power_levels)
                if node.remaining_energy < uav_energy_threshold:
                    for move_idx in range(1, num_move_actions):
                        for power_idx in range(num_power_levels):
                            action_id = move_idx * num_power_levels + power_idx
                            avail_agent_actions[action_id] = 0
            else:
                num_move_actions_ugv = config.ACTION_MOVE_LEVELS_UGV
                num_total_actions_ugv = num_move_actions_ugv * num_power_levels
                avail_agent_actions = [1] * num_total_actions_ugv

                max_action_len_uav = config.ACTION_MOVE_LEVELS_UAV * num_power_levels
                padding_len = max_action_len_uav - num_total_actions_ugv
                if padding_len > 0:
                    avail_agent_actions.extend([0] * padding_len)
                # ----------------------
            avail_actions.append(avail_agent_actions)
        return avail_actions

    def _calculate_reward(self):
        throughput = 0.0
        if self.G.number_of_edges() > 0:
            adj = self.last_adjacency_matrix
            sinr = self.last_sinr_matrix
            degrees = np.sum(self.last_adjacency_matrix, axis=1) + np.sum(self.last_adjacency_matrix, axis=0)

            degrees[degrees == 0] = 1

            bandwidth_per_link_matrix_sender = config.BANDWIDTH / degrees[:, np.newaxis]
            bandwidth_per_link_matrix_receiver = config.BANDWIDTH / degrees[np.newaxis, :]

            rate_matrix_ij = bandwidth_per_link_matrix_sender * np.log2(1 + sinr)
            rate_matrix_ji = bandwidth_per_link_matrix_receiver * np.log2(1 + sinr.T)

            total_rate_matrix = rate_matrix_ij + rate_matrix_ji
            throughput = np.sum(total_rate_matrix * adj)

        norm_throughput = np.tanh(throughput / 1e9)

        connectivity = self._calculate_natural_connectivity(self.G)

        norm_connectivity = np.tanh(connectivity / 10.0)

        total_initial_energy = self.num_uavs * config.UAV_INITIAL_ENERGY
        current_total_energy = np.sum(self.remaining_energies)
        energy_consumed_this_step = self.last_step_total_energy - current_total_energy
        self.last_step_total_energy = current_total_energy

        norm_energy_cost_this_step = energy_consumed_this_step / (total_initial_energy + 1e-6)

        weights = config.CURRENT_REWARD_WEIGHTS
        reward_for_training = (weights["throughput"] * norm_throughput +
                               weights["connectivity"] * norm_connectivity -
                               weights["energy"] * norm_energy_cost_this_step)

        reward_components_for_analysis = {
            'throughput': norm_throughput,
            'connectivity': norm_connectivity,
            'energy_cost': -norm_energy_cost_this_step,
        }

        return reward_for_training, reward_components_for_analysis

    def get_current_performance_metrics(self):
        connectivity_metric = self._calculate_natural_connectivity(self.G)

        throughput = 0.0
        if self.G.number_of_edges() > 0:
            adj = self.last_adjacency_matrix
            sinr = self.last_sinr_matrix
            degrees = np.sum(adj, axis=1) + np.sum(adj, axis=0)
            degrees[degrees == 0] = 1

            bw_sender = config.BANDWIDTH / degrees[:, np.newaxis]
            bw_receiver = config.BANDWIDTH / degrees[np.newaxis, :]

            rate_ij = bw_sender * np.log2(1 + sinr)
            rate_ji = bw_receiver * np.log2(1 + sinr.T)

            total_rate_matrix = rate_ij + rate_ji
            throughput = np.sum(total_rate_matrix * adj)

        total_remaining_energy = np.sum(self.remaining_energies)

        return {
            'throughput': throughput,
            'connectivity': connectivity_metric,
            'total_remaining_energy': total_remaining_energy,
            'num_nodes': self.G.number_of_nodes()
        }

    def _calculate_natural_connectivity(self, G):
        num_nodes = G.number_of_nodes()
        if num_nodes <= 1:
            return 0.0

        try:
            adj_matrix = nx.to_numpy_array(G)

            eigenvalues = np.linalg.eigvalsh(adj_matrix)

            sum_of_exp_eigenvalues = np.sum(np.exp(eigenvalues))

            average_exp = sum_of_exp_eigenvalues / num_nodes

            natural_connectivity = np.log(average_exp + 1e-9)

            return natural_connectivity
        except Exception as e:
            return 0.0