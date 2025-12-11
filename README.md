# 🚀 Resilient Air-Ground Networking via STGAT-MAPPO

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange?logo=pytorch)![License](https://img.shields.io/badge/License-MIT-green)

This project implements a **Resilient Heterogeneous Air-Ground Network** framework using Multi-Agent Reinforcement Learning. It features a novel **STGAT-MAPPO** algorithm that combines Spatio-Temporal Graph Attention Networks with Multi-Agent PPO to maintain connectivity and throughput under dynamic environments and adversarial attacks.

---

### 📂 Project Structure

```
/
├── environment/ # Simulation environment (Physics, UAVs, UGVs, Channel Models)
├── marl/ # Algorithm implementations (Agents, Networks, Buffers)
│
├── config.py # ⚙️ Main configuration (Env parameters, Hyperparameters)
├── config_qmix.py # ⚙️ Specific configuration overrides for QMIX
├── generate_scenarios.py # 🛠 Utility to generate fixed evaluation scenarios
│
├── train.py # 🔥 Train MAIN Model (STGAT-MAPPO)
│
├── train_mlp.py # 📉 Train Baseline: MLP-MAPPO
├── train_ippo.py # 📉 Train Baseline: IPPO
├── train_maddpg.py # 📉 Train Baseline: MADDPG
├── train_qmix.py # 📉 Train Baseline: QMIX
│
├── train_ablation1.py # 🧪 Train Ablation: S-GAT (No Time)
├── train_ablation2.py # 🧪 Train Ablation: ST-GCN (GCN+GRU)
├── train_ablation4.py # 🧪 Train Ablation: S-GCN (GCN only)
├── train_ablation5.py # 🧪 Train Ablation: No Heterogeneity Features
│
├── evaluate.py # 📊 Evaluate a single trained model
├── run_comparison.py # 🆚 Compare multiple models (Baselines or Ablations)
├── run_node_attack.py # 💥 Test resilience against node destruction
│
├── logger.py # Logging utility
├── plotter.py # Plotting utility
└── reward_normalizer.py # Reward normalization utility
```

---

### ⚡️ Quick Start

#### 1. Installation
Ensure you have Python 3.8+ installed. Install PyTorch and PyTorch Geometric according to your CUDA version, then install the rest:
```bash
pip install numpy networkx matplotlib pandas
```

#### 2. Generate Scenarios (Important!)
Before running comparisons or attacks, generate a fixed set of test scenarios to ensure fair evaluation:
```bash
python generate_scenarios.py
```

#### 3. Training Models

**👉 Train Our Proposed Method (STGAT-MAPPO):**
```bash
python train.py
```

**👉 Train Baselines:**
```bash
python train_mlp.py # MLP-MAPPO (No Graph Structure)
python train_ippo.py # IPPO (No Centralized Critic)
python train_maddpg.py # MADDPG (Actor-Critic)
python train_qmix.py # QMIX (Value-Based)
```

**👉 Train Ablation Studies:**
```bash
python train_ablation1.py # Ablation 1: Remove Temporal (GRU)
python train_ablation2.py # Ablation 2: GAT -> GCN
python train_ablation4.py # Ablation 4: GAT+GRU -> GCN only
python train_ablation5.py # Ablation 5: Remove Agent Type Features
```

*Note: Logs will be saved to `logs/` (or `logs_mlp`, `logs_qmix`, etc.) and models to `models/` automatically during runtime.*

#### 4. Visualization
Plot training curves from the generated CSV logs:
```bash
python plotter.py
# Note: You may need to edit the default log directory path in the script to plot different models.
```

---

### 📊 Evaluation & Comparison

#### Compare Performance
Use `run_comparison.py` to benchmark different agents. You can toggle between `BASELINE` mode and `ABLATION` mode inside the script.
```bash
python run_comparison.py
```
*Outputs: Excel report (`.xlsx`) and Pickle file (`.pkl`) with statistical results.*

#### Resilience Test (Node Attack)
Simulate a scenario where the most critical node is destroyed mid-operation to test self-healing capabilities:
```bash
python run_node_attack.py
```

#### Single Model Evaluation
To inspect a specific model checkpoint:
1. Modify `evaluate.py` to point to your desired model path and episode.
2. Run:
```bash
python evaluate.py
```

---

### 📝 Configuration
* **`config.py`**: Controls global settings like `NUM_UAVS`, `NUM_UGVS`, `SCENARIO_TYPE` (Environmental/Adversarial), and PPO hyperparameters.
* **`config_qmix.py`**: Overrides specific parameters (epsilon greedy, buffer size) for the QMIX algorithm.

---

### 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
