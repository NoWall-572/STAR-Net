# environment/channel_models.py (Optimized Version)

import numpy as np
import config

def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)


def dbi_to_linear(dbi):
    return 10 ** (dbi / 10)


def path_loss_to_gain(loss_db):
    return 10 ** (-loss_db / 10)


def calculate_fspl(distance, frequency):
    safe_distance = np.where(distance <= 0, np.nan, distance)
    fspl = 20 * np.log10(safe_distance) + 20 * np.log10(frequency) - 147.55
    return np.where(distance <= 0, np.inf, fspl)


def get_A2A_path_loss(distances):
    fspl_db = calculate_fspl(distances, config.CARRIER_FREQUENCY)
    rain_attenuation_per_km = config.RAIN_ATTENUATION_A * (config.RAIN_RATE ** config.RAIN_ATTENUATION_B)
    rain_attenuation_db = rain_attenuation_per_km * (distances / 1000.0)
    return fspl_db + rain_attenuation_db


def get_A2G_path_loss(uav_pos_arr, ugv_pos_arr):
    distance_3d = np.linalg.norm(uav_pos_arr - ugv_pos_arr, axis=1)
    d_horizontal = np.linalg.norm(uav_pos_arr[:, :2] - ugv_pos_arr[..., :2], axis=1)
    h_uav = uav_pos_arr[:, 2]

    elevation_rad = np.arctan2(h_uav, d_horizontal)
    elevation_deg = np.degrees(elevation_rad)

    a, b = config.A2G_LOS_PROB_A, config.A2G_LOS_PROB_B
    prob_los = 1.0 / (1.0 + a * np.exp(-b * (elevation_deg - a)))

    num_links = uav_pos_arr.shape[0]
    is_los = np.random.rand(num_links) < prob_los

    fspl_db = calculate_fspl(distance_3d, config.CARRIER_FREQUENCY)

    return np.where(is_los, fspl_db, fspl_db + config.A2G_NLOS_EXTRA_LOSS)