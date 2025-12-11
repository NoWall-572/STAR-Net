# 🌍 Environment Module

This directory contains the core components for simulating the heterogeneous Air-Ground network. It handles the physics, agent dynamics, and environmental interactions.

### 📂 File Descriptions

*   **`env.py`**
    *   The main simulation class `AirGroundEnv`. It manages the state transitions, calculates the dynamic network topology based on SINR, computes rewards (throughput, connectivity, energy), and provides observations to the agents.

*   **`entities.py`**
    *   Defines the physical agents within the simulation.
    *   **`UAV`**: Aerial agents with 3D mobility constraints and complex energy consumption models (flight + communication).
    *   **`UGV`**: Ground agents with 2D mobility and higher payload/power capabilities.

*   **`channel_models.py`**
    *   Implements realistic wireless communication physics.
    *   Includes vectorized calculations for **Path Loss** (Free Space, A2A, A2G, G2G) and fading effects (Rician fading, Rain attenuation).

*   **`threats.py`**
    *   Models external factors affecting the network.
    *   **`Jammer`**: Simulates adversarial jamming attacks.
    *   **`EnvironmentalThreats`**: Simulates natural conditions like wind fields affecting UAV energy.
