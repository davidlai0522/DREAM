# DREAM ​- Domain Reasoning with Reinforcement ​for Effective Arm Manipulation​

A Python package that integrates PDDL (Planning Domain Definition Language) task planning with reinforcement learning for robotic control and execution. This is a coursework project for CS5446 AI Planning and Decision Making.

## Overview

This package provides a framework for:
1. Defining and solving PDDL planning problems for robotic tasks
2. Translating high-level PDDL plans into executable robot actions
3. Using reinforcement learning policies to execute these actions in simulation or on real robots
4. Monitoring execution and adapting to environmental changes

## Features

- **PDDL Planning**: Solve planning problems using various search algorithms
- **RL Policy Integration**: Execute plans using pre-trained reinforcement learning policies
- **Modular Architecture**: Easily extend with new planners, policies, or robot interfaces

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pddl_rl_robot.git
cd pddl_rl_robot

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Dependencies

Please refer to the `requirements.txt` file for a list of all dependencies.

## Usage

### Basic Usage

This section will be made available soon.
```bash
python3 main.py # For final integration
```

## Project Structure

```
pddl_rl_robot/
├── config/                 # Configuration files
│   └── project_config.yaml # Main configuration
│   └── domain.pddl         # PDDL domain definition
│   └── problem.pddl        # PDDL problem definition
├── examples/               # Example scripts and usage demonstrations
├── pddl_rl_robot/          # Main package
│   ├── deterministic/      # Deterministic policy
│   ├── pddl/               # PDDL parsing and solving
│   ├── rl/                 # RL policy management
│   ├── simulation/         # Simulation of environments
│   └── utils/              # General utility functions
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── main.py                 # Main entry point for integration
```

# Demo
Click the image below to watch a demo of the DREAM framework in action.
[![DREAM Demo](https://img.youtube.com/vi/HSHSuAT0FLc/0.jpg)](https://youtu.be/HSHSuAT0FLc)


## Maintainer

- [Joseph Phang](https://github.com/PDYJJJ)
- [Edward Xiao](https://github.com/yxiaoaz)
- [David Lai](https://github.com/davidlai0522)
- [Toh Hoon Chew](https://github.com/hoonchew)
