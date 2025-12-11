# 🚀 Resilient Air-Ground Networking with STGAT-MAPPO

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange?logo=pytorch)![License](https://img.shields.io/badge/License-MIT-green)

This project presents a framework for simulating and controlling a resilient, heterogeneous air-ground network using Multi-Agent Reinforcement Learning (MARL). The core of our solution is **STGAT-MAPPO**, an algorithm where each agent (UAV or UGV) uses a Spatio-Temporal Graph Attention Network (STGAT) to model the complex network dynamics and learns collaborative policies via Multi-Agent Proximal Policy Optimization (MAPPO).

---

### ✅ Key Features

* **Heterogeneous Multi-Agent System**: Simulates a network of both Unmanned Aerial Vehicles (UAVs) and Unmanned Ground Vehicles (UGVs), each with distinct physical and communication characteristics.
* **Dynamic Network Topology**: The communication graph evolves based on agent mobility, signal propagation physics, and potential link disruptions.
* **Graph-Based MARL**: Employs a powerful STGAT encoder to process local network topology and learn predictive, robust communication strategies.
* **Adversarial & Environmental Scenarios**: The environment, configurable in `config.py`, can simulate threats like jammers, wind, and rain to test network resilience.
* **Comprehensive Evaluation Suite**: Includes scripts for training, plotting results, evaluating final models, comparing different agents, and running specific attack simulations.

---

### 📂 Project Structure

```
/
├── environment/ # Simulation environment module
│ ├── env.py # Main environment class
│ ├── entities.py # UAV and UGV classes
│ ├── channel_models.py # Wireless propagation physics
│ └── threats.py # Jammer and environmental threats
├── marl/ # MARL algorithm module
│ ├── mappo_agent.py # Main STGAT-MAPPO agent
│ ├── networks.py # Main network architectures (STGAT, Actor, Critic)
│ ├── buffer.py # On-policy replay buffer
│ ├── ..._ablationX.py # Agents and networks for ablation studies
│ └── ..._baseline.py # Agents and networks for baseline models
├── logs/ # Directory for training logs (.csv files)
├── models/ # Directory for saved model weights (.pth files)
├── config.py # 📜 Central configuration file for all parameters
├── train.py # ⚡️ Main training script for the STGAT-MAPPO model
├── train_ablation1.py # Training script for S-GAT-MAPPO
├── train_ablation2.py # Training script for ST-GCN-MAPPO
├── train_ablation3_mlp.py # Training script for MLP-MAPPO (baseline)
├── train_ablation4.py # Training script for S-GCN-MAPPO
├── train_ablation5.py # Training script for STGAT-MAPPO without heterogeneity features
├── evaluate.py # 📊 Script to evaluate a trained model
├── run_comparison.py # 🤖 Script to compare performance across different agents
├── run_node_attack.py # 💥 Script to simulate a targeted node attack and test self-healing
└── plotter.py # 📈 Script to generate performance plots from log files
```

---

### ⚙️ Setup

1. **Clone the repository:**


2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

3. **Install dependencies:**
This project requires PyTorch and PyTorch Geometric. Please follow their official installation instructions first to ensure compatibility with your CUDA version.
* [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
* [PyTorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

Then, install the remaining packages:
```bash
pip install numpy networkx matplotlib pandas
```

---

### ⚡️ Usage Guide

#### 1. Configuration

Before running any script, review **`config.py`**. This is the central hub for tuning the simulation and training parameters. You can easily switch between scenarios, change the number of agents, adjust reward weights, and modify hyperparameters.

```python
# In config.py
SCENARIO_TYPE = 'ENVIRONMENTAL' # or 'ADVERSARIAL'
NUM_UAVS = 8
NUM_UGVS = 4
CURRENT_REWARD_WEIGHTS = REWARD_WEIGHTS_LOITERING # or REWARD_WEIGHTS_RECON, etc.
```

#### 2. Training

All training scripts leverage multiprocessing for efficient data collection. They will automatically create new directories for logs and models (e.g., `logs_ablation1/`, `models_ablation1/`) to avoid conflicts.

* **Train the main STGAT-MAPPO model:**
```bash
python train.py
```

* **Run Ablation Studies:**
```bash
# Ablation 1: S-GAT-MAPPO (Removes GRU)
python train_ablation1.py

# Ablation 2: ST-GCN-MAPPO (Replaces GAT with GCN, keeps GRU)
python train_ablation2.py

# Ablation 4: S-GCN-MAPPO (Replaces GAT with GCN, removes GRU)
python train_ablation4.py

# Ablation 5: STGAT-MAPPO (Removes agent type features)
python train_ablation5.py
```

* **Run Baseline Models:**
```bash
# Baseline: MLP-MAPPO (Removes all graph structure)
python train_ablation3_mlp.py
```

#### 3. Plotting Training Progress

To visualize the results, run the plotter script. It automatically finds the latest `.csv` log in the specified directory and saves performance plots as `.png` images. The script has been updated to save files instead of displaying them, making it suitable for server use.

```bash
# To plot results from the main training run
python plotter.py
# (It will check the default 'logs' directory)
```
*Note: You may need to edit the default path in `plotter.py` if you wish to plot logs from ablation directories like `logs_ablation1`.*

#### 4. Evaluating a Trained Model

After training, you can evaluate the performance of a specific saved model. Open `evaluate.py` and set the `MODEL_EPISODE_TO_LOAD` and model directory variables to the desired values.

```python
# In evaluate.py
MODEL_EPISODE_TO_LOAD = 1400 # Change this value
actor_path = f'./models/actor_{MODEL_EPISODE_TO_LOAD}.pth' # Change model directory if needed```
Then run the script:
```bash
python evaluate.py
```

#### 5. Comparing Agents

To run a head-to-head comparison of the final performance of different agents (e.g., STGAT vs. MLP vs. Heuristic), use the `run_comparison.py` script. You can enable/disable agents and set their model paths directly within the file.

```bash
python run_comparison.py
```

#### 6. Simulating a Node Attack

To test the network's resilience and self-healing capabilities, use the `run_node_attack.py` script. It runs the simulation until a stable topology is formed, identifies and removes the most critical node, and observes the network's recovery. You can select the agent to test inside the script.

```bash
python run_node_attack.py
```

---

### 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
