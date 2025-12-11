# config_qmix.py
from config import *
import torch

TOTAL_EPISODES = 20000
BUFFER_CAPACITY = 5000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 100
TRAIN_STEPS_PER_EPISODE = 8
SAVE_INTERVAL = 500

EPSILON_START = 1.0
EPSILON_FINISH = 0.05
EPSILON_DECAY_STEPS = 150000

LEARNING_RATE = 5e-5

print("--- The QMIX dedicated configuration file (config_qmix.py) has been loaded. ---")