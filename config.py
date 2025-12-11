# config.py

import numpy as np
import torch


# 'ENVIRONMENTAL': Simulating the effects of natural environments
# 'ADVERSARIAL': Simulating human-environment interaction
SCENARIO_TYPE = 'ADVERSARIAL'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if SCENARIO_TYPE == 'ENVIRONMENTAL':
    NUM_UAVS = 8
    NUM_UGVS = 4
    AREA_SIZE = 1000.0
    WIND_VECTOR = np.array([6.0, -6.0, 0.0])
    RAIN_RATE = 5.0
    JAMMER_POWER_dBm = -999.0
    JAMMER_GAIN_dBi = 0.0
    JAMMER_POSITION = None
else:
    NUM_UAVS = 5
    NUM_UGVS = 3
    AREA_SIZE = 500.0
    WIND_VECTOR = np.array([0.0, 0.0, 0.0])
    RAIN_RATE = 0.0
    JAMMER_POWER_dBm = 10.0
    JAMMER_GAIN_dBi = 10.0
    JAMMER_POSITION = None

SIM_TIME_STEP = 0.1
EPISODE_LENGTH = 100

UAV_MAX_SPEED = 20.0 # m/s
UAV_MAX_ACCELERATION = 5.0 # m/s^2
UAV_INITIAL_ENERGY = 100000.0
UGV_MAX_SPEED = 10.0
UGV_MAX_ACCELERATION = 3.0

NODE_TRANSMIT_POWER_dBm = 23.0
CARRIER_FREQUENCY = 2.4e9
LIGHT_SPEED = 3.0e8
NOISE_POWER_dBm = -90.0
BANDWIDTH = 1e7
SINR_THRESHOLD_dB = 3.0

RAIN_ATTENUATION_A = 0.012
RAIN_ATTENUATION_B = 1.2
A2G_LOS_PROB_A = 9.61
A2G_LOS_PROB_B = 0.16
A2G_NLOS_EXTRA_LOSS = 20.0
G2G_PATH_LOSS_EXPONENT_N = 3.0
G2G_PATH_LOSS_CONSTANT_C = 40.0
G2G_RICIAN_K_FACTOR = 5.0

UAV_P_BASE_COEFF = 0.5
UAV_P_WIND_COEFF = 1.0
UAV_P_COMM_COEFF = 2.0


ACTION_POWER_LEVELS = 5
ACTION_MOVE_LEVELS_UAV = 7
ACTION_MOVE_LEVELS_UGV = 5

NODE_FEATURE_DIM = 11 + 2 + 1

GNN_EMBEDDING_DIM = 64
RNN_HIDDEN_DIM = 64
ACTOR_HIDDEN_DIM = 64
CRITIC_HIDDEN_DIM = 128

LEARNING_RATE = 3e-5
GAMMA = 0.99
PPO_EPOCHS = 10
PPO_CLIP_PARAM = 0.1
ENTROPY_COEFF = 0.08
GAE_LAMBDA = 0.95
UPDATE_TIMESTEPS = EPISODE_LENGTH * 8

MAX_OBS_NODES = 20



# REWARD WEIGHTS

# Reconnaissance
REWARD_WEIGHTS_RECON = {
    "throughput": 0.6, "connectivity": 0.2, "energy": 0.2
}
# Command Chain
REWARD_WEIGHTS_COMMAND = {
    "throughput": 0.1, "connectivity": 0.7, "energy": 0.2
}
# Stealth & Loitering
REWARD_WEIGHTS_LOITERING = {
    "throughput": 0.2, "connectivity": 0.2, "energy": 0.6
}


CURRENT_REWARD_WEIGHTS = REWARD_WEIGHTS_LOITERING








