# 🧠 MARL Algorithms & Networks

This directory houses the implementations of the Reinforcement Learning agents, neural network architectures, and experience replay buffers.

### 🌟 Our Method (STGAT-MAPPO)
*   **`mappo_agent.py`**: The core agent implementation using Multi-Agent PPO with centralized training.
*   **`networks.py`**: Defines the **STGATEncoder** (Spatio-Temporal Graph Attention Network) and the Actor/Centralized Critic networks.
*   **`buffer.py`**: An on-policy replay buffer designed for PPO updates.

### 📉 Baselines
We implement several SOTA benchmarks for comparison:

1.  **MLP-MAPPO (No Graph Structure)**
    *   `mappo_agent_mlp.py`: MAPPO agent using simple MLP encoders.
    *   `networks_baseline.py`: Defines the `MLPEncoder` which treats observations as flat vectors.

2.  **QMIX (Value-Based)**
    *   `qmix_agent.py`: Implementation of the QMIX algorithm.
    *   `networks_qmix.py`: Contains the `QMIX_RNNAgent` and the `MixingNetwork`.
    *   `replay_buffer.py`: An off-policy episodic replay buffer for QMIX.

3.  **IPPO (Independent PPO)**
    *   `ippo_agent.py`: Independent PPO agents with no centralized critic.
    *   `networks_ippo.py`: Defines the `DecentralizedCritic`.

4.  **MADDPG (Actor-Critic)**
    *   `maddpg_agent.py`: Multi-Agent DDPG implementation with Gumbel-Softmax for discrete actions.
    *   `networks_maddpg.py`: Actor and Critic networks for MADDPG.
    *   `buffer_maddpg.py`: Replay buffer tailored for MADDPG.

### 🧪 Ablation Studies
Variants of our main model to test specific components:

*   **Ablation 1 (No Temporal Modeling):**
    *   `mappo_agent_ablation1.py` & `networks_ablation1.py`: Uses **S-GAT** (Spatial GAT only), removing the GRU.
*   **Ablation 2 (No Attention Mechanism):**
    *   `mappo_agent_ablation2.py` & `networks_ablation2.py`: Replaces GAT with **GCN** (Graph Convolution), keeping the GRU.
*   **Ablation 4 (No Temporal & No Attention):**
    *   `mappo_agent_ablation4.py` & `networks_ablation4.py`: Uses **S-GCN** (Spatial GCN only), removing both GAT and GRU.
*   **Ablation 5 (No Heterogeneity Features):**
    *   Uses the standard `mappo_agent.py` but is trained with input features that exclude agent type indicators.
