# generate_scenarios.py

import pickle
import numpy as np
from environment.env import AirGroundEnv
import config

NUM_SCENARIOS = 200
OUTPUT_FILE = 'evaluation_scenarios.pkl'


def generate_and_save_scenarios():
    print(f"--- Generating {NUM_SCENARIOS} evaluation scenarios ---")

    env = AirGroundEnv()

    scenarios = []

    for i in range(NUM_SCENARIOS):
        env.reset()

        uav_positions = [uav.initial_position for uav in env.uavs]
        ugv_positions = [ugv.initial_position for ugv in env.ugvs]

        scenario_config = {
            'uav_positions': uav_positions,
            'ugv_positions': ugv_positions
        }
        scenarios.append(scenario_config)

        print(f"{i + 1}/{NUM_SCENARIOS} scenes generated...")

    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(scenarios, f)

    print(f"\n--- Success! All scenes have been saved to the file: {OUTPUT_FILE} ---")


if __name__ == '__main__':
    print(f"A scene will be generated for {config.NUM_UAVS} UAVs and {config.NUM_UGVS} UGVs.")
    generate_and_save_scenarios()