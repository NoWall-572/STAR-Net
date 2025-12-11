# logger.py

import csv
import os
import datetime

class Logger:
    def __init__(self, log_dir='logs'):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename = os.path.join(log_dir, f"training_log_{current_time}.csv")
        self.file = open(self.filename, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            'Episode', 'Total_Steps',
            'Raw_Throughput', 'Raw_Connectivity', 'Raw_EnergyCost',
            'Actor_Loss', 'Critic_Loss', 'Entropy', 'Explained_Variance'
        ])
        self.file.flush()
        print(f"Logs will be recorded to: {self.filename}")

    def log(self, episode, total_steps, reward_components, diagnostics):
        self.writer.writerow([
            episode, total_steps,
            reward_components.get('throughput', 0),
            reward_components.get('connectivity', 0),
            reward_components.get('energy_cost', 0),
            diagnostics.get('actor_loss', 0),
            diagnostics.get('critic_loss', 0),
            diagnostics.get('entropy', 0),
            diagnostics.get('explained_variance', 0)
        ])
        self.file.flush()

    def close(self):
        self.file.close()