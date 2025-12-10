# environment/entities.py

import numpy as np
import config


class BaseNode:
    def __init__(self, node_id, initial_position, initial_velocity, max_speed, max_acceleration):
        self.id = node_id
        self.initial_position = np.array(initial_position, dtype=float)
        self.initial_velocity = np.array(initial_velocity, dtype=float)
        self.pos = np.copy(self.initial_position)
        self.vel = np.copy(self.initial_velocity)
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration

    def update_state(self, acceleration_command):
        acc_command = np.array(acceleration_command, dtype=float)
        command_norm = np.linalg.norm(acc_command)
        if command_norm > self.max_acceleration:
            acc_command = acc_command * (self.max_acceleration / command_norm)

        self.vel += acc_command * config.SIM_TIME_STEP
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel * (self.max_speed / speed)

        self.pos += self.vel * config.SIM_TIME_STEP

    def reset(self):
        self.pos = np.copy(self.initial_position)
        self.vel = np.copy(self.initial_velocity)


class UAV(BaseNode):
    def __init__(self, node_id, initial_position):
        super().__init__(node_id, initial_position, [0.0, 0.0, 0.0],
                         config.UAV_MAX_SPEED, config.UAV_MAX_ACCELERATION)
        self.initial_energy = config.UAV_INITIAL_ENERGY
        self.remaining_energy = self.initial_energy

    def update_energy_consumption(self, wind_vector, transmit_power_watts):
        p_base = config.UAV_P_BASE_COEFF * (np.linalg.norm(self.vel) ** 3)
        relative_wind_speed = np.linalg.norm(self.vel - wind_vector)
        p_wind = config.UAV_P_WIND_COEFF * (relative_wind_speed ** 2)
        p_prop = p_base + p_wind
        p_comm = config.UAV_P_COMM_COEFF * transmit_power_watts
        total_power = p_prop + p_comm
        energy_consumed = total_power * config.SIM_TIME_STEP
        self.remaining_energy -= energy_consumed
        if self.remaining_energy < 0:
            self.remaining_energy = 0

    def reset(self):
        super().reset()
        self.remaining_energy = self.initial_energy


class UGV(BaseNode):
    def __init__(self, node_id, initial_position):
        initial_position_3d = list(initial_position)[:2] + [0.0]
        super().__init__(node_id, initial_position_3d, [0.0, 0.0, 0.0],
                         config.UGV_MAX_SPEED, config.UGV_MAX_ACCELERATION)

    def update_state(self, acceleration_command):
        acc_command_3d = list(acceleration_command)[:2] + [0.0]
        super().update_state(acc_command_3d)
        self.pos[2] = 0.0
        self.vel[2] = 0.0