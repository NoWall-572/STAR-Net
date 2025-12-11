# plotter.py

import os

import matplotlib
matplotlib.use('Agg')
# -------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_all_metrics(log_dir='logs', window_size=10):
    try:
        list_of_files = glob.glob(os.path.join(log_dir, '*.csv'))
        if not list_of_files:
            print(f"Error: No .csv log files found in the ‘{log_dir}’ directory.")
            return

        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Reading log file: {latest_file}")

        base_filename, _ = os.path.splitext(latest_file)
        performance_plot_path = f"{base_filename}_performance.png"
        diagnostics_plot_path = f"{base_filename}_diagnostics.png"
        # ----------------------------------------------------

        data = pd.read_csv(latest_file)

        metrics_to_smooth = [
            'Raw_Throughput', 'Raw_Connectivity', 'Raw_EnergyCost',
            'Actor_Loss', 'Critic_Loss', 'Entropy', 'Explained_Variance'
        ]
        for metric in metrics_to_smooth:
            if metric in data.columns:
                data[f'Smoothed_{metric}'] = data[metric].rolling(window=window_size, min_periods=1).mean()

        plt.style.use('seaborn-v0_8-whitegrid')

        fig1, axs1 = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        fig1.suptitle('Training Progress: Core Performance Metrics', fontsize=18)

        axs1[0].plot(data['Episode'], data['Smoothed_Raw_Throughput'], color='forestgreen', linewidth=2)
        axs1[0].set_title('Smoothed Throughput (Higher is Better)')
        axs1[0].set_ylabel('Throughput')
        axs1[0].scatter(data['Episode'], data['Raw_Throughput'], alpha=0.1, s=5, color='forestgreen')

        axs1[1].plot(data['Episode'], data['Smoothed_Raw_Connectivity'], color='darkorange', linewidth=2)
        axs1[1].set_title('Smoothed Natural Connectivity (Higher is Better)')
        axs1[1].set_ylabel('Natural Connectivity')
        axs1[1].scatter(data['Episode'], data['Raw_Connectivity'], alpha=0.1, s=5, color='darkorange')

        axs1[2].plot(data['Episode'], data['Smoothed_Raw_EnergyCost'], color='crimson', linewidth=2)
        axs1[2].set_title('Smoothed Energy Cost (Closer to Zero is Better)')
        axs1[2].set_ylabel('Energy Cost')
        axs1[2].scatter(data['Episode'], data['Raw_EnergyCost'], alpha=0.1, s=5, color='crimson')

        axs1[2].set_xlabel('Episode')
        for ax in axs1:
            ax.grid(True)
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig1.savefig(performance_plot_path)
        print(f"The core performance metrics chart has been saved to: {performance_plot_path}")
        # -----------------------------------

        fig2, axs2 = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
        fig2.suptitle('Training Progress: Diagnostic Metrics', fontsize=18)

        axs2[0].plot(data['Episode'], data['Smoothed_Critic_Loss'], color='royalblue', linewidth=2)
        axs2[0].set_title('Smoothed Critic Loss (Should decrease and stabilize at a low value)')
        axs2[0].set_ylabel('Critic Loss')
        axs2[0].scatter(data['Episode'], data['Critic_Loss'], alpha=0.1, s=5, color='royalblue')

        axs2[1].plot(data['Episode'], data['Smoothed_Entropy'], color='mediumorchid', linewidth=2)
        axs2[1].set_title('Smoothed Policy Entropy (Should decrease and stabilize)')
        axs2[1].set_ylabel('Entropy')
        axs2[1].scatter(data['Episode'], data['Entropy'], alpha=0.1, s=5, color='mediumorchid')

        axs2[2].plot(data['Episode'], data['Smoothed_Explained_Variance'], color='teal', linewidth=2)
        axs2[2].set_title('Smoothed Explained Variance (Should rise and stabilize near 1.0)')
        axs2[2].set_ylabel('Explained Variance')
        axs2[2].set_ylim(-0.1, 1.1)
        axs2[2].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axs2[2].axhline(y=1, color='gray', linestyle='--', linewidth=1)
        axs2[2].scatter(data['Episode'], data['Explained_Variance'], alpha=0.1, s=5, color='teal')

        axs2[3].plot(data['Episode'], data['Smoothed_Actor_Loss'], color='saddlebrown', linewidth=2)
        axs2[3].set_title('Smoothed Actor Loss (Should oscillate around a stable value)')
        axs2[3].set_ylabel('Actor Loss')
        axs2[3].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axs2[3].scatter(data['Episode'], data['Actor_Loss'], alpha=0.1, s=5, color='saddlebrown')

        axs2[3].set_xlabel('Episode')
        for ax in axs2:
            ax.grid(True)
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig2.savefig(diagnostics_plot_path)
        print(f"Internal diagnostic indicator chart saved to: {diagnostics_plot_path}")

        plt.close(fig1)
        plt.close(fig2)
        print("Drawing complete.")
        # ------------------------------------

    except Exception as e:
        print(f"An error occurred while drawing the chart: {e}")
        print("Please verify that the column names in the CSV file exactly match the names used in the code (‘Raw_Throughput’, ‘Critic_Loss’, etc.).")


if __name__ == '__main__':
    plot_all_metrics()