# DREAM Tutorial: Domain Reasoning with Reinforcement for Effective Arm Manipulation

Welcome to the tutorial for the DREAM project! This guide will walk you through the setup, usage, and customization of the DREAM framework, which integrates PDDL task planning with reinforcement learning for robotic control.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Basic Usage](#basic-usage)
4. [Understanding the Project Structure](#understanding-the-project-structure)
5. [Customizing the Framework](#customizing-the-framework)
6. [Running Examples](#running-examples)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

DREAM is a Python package designed for robotic task planning and execution. It combines **PDDL (Planning Domain Definition Language)** for high-level task planning with **Reinforcement Learning (RL)** for low-level control. This framework is modular, allowing easy integration of new planners, policies, or robot interfaces.

---

## Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- `pip` for managing Python packages
- MuJoCo for simulation ([installation guide](https://mujoco.org/))

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pddl_rl_robot.git
   cd pddl_rl_robot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

---

## Basic Usage

### Running the Main Script

To execute the integrated PDDL and RL pipeline:
```bash
python3 main.py
```

This script:
1. Loads the configuration from `config/project_config.yaml`.
2. Solves a PDDL planning problem.
3. Executes the plan using RL policies in simulation.

---

## Understanding the Project Structure

The project is organized as follows:

```
pddl_rl_robot/
├── config/                 # Configuration files
│   ├── project_config.yaml # Main configuration
│   ├── domain.pddl         # PDDL domain definition
│   └── problem.pddl        # PDDL problem definition
├── examples/               # Example scripts and usage demonstrations
├── pddl_rl_robot/          # Main package
│   ├── deterministic/      # Deterministic policy
│   ├── pddl/               # PDDL parsing and solving
│   ├── rl/                 # RL policy management
│   │   ├── move_action/    # RL policies for movement
│   │   ├── grasp_action/   # RL policies for grasping
│   │   └── place_action/   # RL policies for placing
│   ├── simulation/         # Simulation environments and controllers
│   └── utils/              # General utility functions
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── main.py                 # Main entry point for integration
└── tests/                  # Unit and integration tests
```

---

## Customizing the Framework

### Modifying PDDL Files

- **Domain File (`config/domain.pddl`)**: Defines the actions and predicates for the task.
- **Problem File (`config/problem.pddl`)**: Specifies the initial state and goal.

### Adding a New RL Policy

1. Create a new folder under `pddl_rl_robot/rl/` (e.g., `new_action/`).
2. Implement the environment and training logic.
3. Update `config/project_config.yaml` to include the new policy.

---

## Running Examples

### Example: Loading an Environment

Run the example script to load and interact with a simulation environment:
```bash
python3 examples/load_environment.py
```

### Example: Training an RL Model

Train a reinforcement learning model using the provided scripts:
```bash
python3 examples/train_rl_model.py
```

---

## Troubleshooting

### Common Issues

1. **MuJoCo Errors**: Ensure MuJoCo is installed and licensed correctly.
2. **Dependency Issues**: Verify all dependencies are installed using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Simulation Crashes**: Check the configuration in `config/project_config.yaml` for invalid parameters.

---

For further assistance, refer to the [README.md](README.md) or contact the maintainers.

Happy coding!