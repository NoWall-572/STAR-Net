# 📦 Environment Module

This directory contains all the components required to build and simulate the dynamic, heterogeneous air-ground network environment. Each file defines a core aspect of the simulation physics and logic.

---

### Core Components

*   📄 **`env.py`**  
    This is the main environment class (`AirGroundEnv`). It orchestrates the entire simulation, managing agent states, updating the network graph based on communication physics, calculating rewards, and providing observations to the MARL agents.

*   📄 **`entities.py`**  
    Defines the physical agents in the simulation. It contains the base class `BaseNode` and specific implementations for `UAV` (Unmanned Aerial Vehicle) and `UGV` (Unmanned Ground Vehicle), including their unique properties like energy constraints and movement dynamics.

*   📄 **`channel_models.py`**  
    Implements the wireless communication channel physics. This includes vectorized functions for calculating Free Space Path Loss (FSPL), Air-to-Air (A2A), Air-to-Ground (A2G), and Ground-to-Ground (G2G) path loss, incorporating effects like Rician fading and rain attenuation.

*   📄 **`threats.py`**  
    Defines adversarial and environmental threats. This includes the `Jammer` class, which can apply targeted interference, and the `EnvironmentalThreats` class, which simulates effects like wind that impact UAV energy consumption.
