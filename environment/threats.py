# environment/threats.py (Optimized Version)

import numpy as np
import config
from . import channel_models


class EnvironmentalThreats:
    def __init__(self):
        self.wind_vector = config.WIND_VECTOR

    def get_wind_vector(self):
        return self.wind_vector


class Jammer:
    def __init__(self, jammer_id, position, power_dbm, gain_dbi):
        self.id = jammer_id
        self.pos = np.array(position) if position is not None else None
        self.power_watts = channel_models.dbm_to_watts(power_dbm)
        self.gain_linear = channel_models.dbi_to_linear(gain_dbi)

    def set_position(self, new_position):
        self.pos = np.array(new_position)

    def get_jamming_interference(self, receiver_pos_array):
        if self.pos is None or self.power_watts < 1e-12:
            return np.zeros(receiver_pos_array.shape[0])
        path_loss_db_array = channel_models.get_A2G_path_loss(receiver_pos_array, self.pos)

        channel_gain_array = channel_models.path_loss_to_gain(path_loss_db_array)

        interference_watts_array = self.power_watts * self.gain_linear * channel_gain_array

        return interference_watts_array