# PDDL-RL Robot Package Documentation
This documentation provides an overview of the entire PDDL-RL Robot package, including its components, functionality, and usage.

## Overview
The PDDL-RL Robot package is a collection of Python modules that provide a interface between a simulated robot and a PDDL planner. The package consists of the following components:
1. Simulator: A simulator for a simulated robot environment.
2. PDDL Solver: A PDDL solver that generates plans for the robot to follow.
3. Reinforcement Learning: A module for training a reinforcement learning agent to control the robot.

## To-Do-List
### PDDL
- [ ] Develop `solver.py` to replace the current placeholder class.
- [ ] Define `domain.pddl` and `problem.pddl` files for the actual scenario.

### Reinforcement Learning
- [ ] Create the appropriate environment (with well designed reward function) for different atomic actions.
  - `pddl_rl_robot/rl/reach_action/reach_training_env.py` is an example of how we can train a PPO for reach action in `ReachTrainingEnv`. Please replace it with the well designed reward functions.
- [ ] Complete the RL training for other actions (lift, pick and place)
- [ ] Create a module for training a reinforcement learning agent to control the robot with different algorithms (PPO, DQN, Actor-Critic etc.).


### Other
- [ ] Improve `main.py` to handle the actual robot control and planning.
