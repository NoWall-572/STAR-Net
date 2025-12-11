# 🧠 MARL Core Module

This directory houses the core components of the Multi-Agent Reinforcement Learning algorithms used in this project. It includes the agent logic, the neural network architectures, and the data storage utilities.

---

### Core Components

*   📄 **`mappo_agent.py`**  
    The primary implementation of our main agent, which uses the **STGAT-MAPPO** algorithm. It defines how the agent chooses actions, processes observations through the encoder, and updates its policies using a centralized critic.

*   📄 **`networks.py`**  
    Contains the neural network architectures for the main STGAT-MAPPO model. This includes the `STGATEncoder` (which combines GAT and GRU), the `Actor` network, and the centralized `Critic` network.

*   📄 **`buffer.py`**  
    Implements the `OnPolicyBuffer`, a replay buffer specifically designed for on-policy algorithms like PPO. It stores trajectories of experience collected by the agents for policy updates.

*   📄 **`mappo_agent_mlp.py` & `networks_baseline.py`**  
    These files define the **MLP-MAPPO** baseline. `MLPEncoder` replaces the GNN with a simple Multi-Layer Perceptron, serving as a key baseline to validate the effectiveness of graph modeling.

*   📄 **`mappo_agent_ablationX.py` & `networks_ablationX.py`**  
    A series of files supporting the ablation studies. Each pair defines a modified agent and/or network architecture to isolate and test the contribution of a specific component of our main model (e.g., removing temporal modeling, replacing GAT with GCN).
