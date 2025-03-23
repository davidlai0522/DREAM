from pddl_rl_robot.pddl.solver import PDDLSolver
from pddl_rl_robot.utils.config_loader import ConfigLoader
from pddl_rl_robot.simulation.simulator import RobotSimulator
import time
import numpy as np
from termcolor import colored
import os
import json
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
from robosuite.wrappers import GymWrapper


"""
This script is the main entry point for the PDDL-RL robot system. The system
consists of a PDDL (Planning Domain Definition Language) task planner and a
reinforcement learning (RL) agent. The PDDL task planner generates a plan,
a sequence of high-level actions that need to be executed in order to achieve
the desired goal. The RL agent then executes the plan, using reinforcement
learning to learn a policy that maps the current state of the robot to a
sequence of low-level actions that can be executed in order to achieve the
desired goal.

The main flow of this script is as follows:

1. Load the config from a YAML file.
2. Connect to the robot using the config.
3. Load the PDDL task planner and reinforcement learning agent using the config.
4. Solve the PDDL problem to get a plan using the config.
5. Execute the plan using the reinforcement learning agent using the config.
"""

def main():
    """Main function."""

    # STEP 0: Initialization
    NUM_STEP = 1000
    simulation = RobotSimulator(
        robot_type="Panda",
        env_name="two_peg_one_disk",
    )
    simulation.reset()

    print(f"The action space of Panda Robot is: {simulation.gym_env.action_space}")
    print(f"The observation space of Panda Robot is: {simulation.gym_env.observation_space}")


    # ====================================== Load config ======================================
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/project_config.yaml")
    print(colored("-" * 50, "blue", attrs=["bold"]))
    print(colored(f"Loading config from {config_path}", "cyan", attrs=["bold"]))

    config_loader = ConfigLoader(config_path)
    print(colored("PDDL config:", "cyan", attrs=["bold"]))
    print(json.dumps(config_loader.get_pddl_config(), indent=4))
    print(colored("Reinforcement learning config:", "cyan", attrs=["bold"]))
    print(json.dumps(config_loader.get_reinforcement_learning_config(), indent=4))

    # Create and initialize the environment
    print(colored("-" * 50, "blue", attrs=["bold"]))
    print(colored("Initializing environment...", "cyan", attrs=["bold"]))

    # ====================================== Solve PDDL problem ======================================
    print(colored("-" * 50, "blue", attrs=["bold"]))

    domain_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", config_loader.get_pddl_config()["domain_file"])
    problem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", config_loader.get_pddl_config()["problem_file"])

    task_planner = PDDLSolver(domain_file=domain_path, problem_file=problem_path)

    print(colored("Solving the PDDL problem...", "cyan", attrs=["bold"]))
    plan = task_planner.solve()
    print(colored(f"Plan: {plan}", "green"))

    # ====================================== Exxecute the plan ======================================
    print(colored("-" * 50, "blue", attrs=["bold"]))
    print(colored("Executing the plan...", "cyan", attrs=["bold"]))

    for idx, step in enumerate(plan):
        step = step.split()
        action = step[0]
        parameters = step[1:]
        print(colored(f"Executing action {idx+1}: {action} with parameters: {parameters}", "green"))

        try:
            for _ in range(NUM_STEP):
                # NOTE: the implementation here is temporary, it should be replaced by the RL's policy.
                if idx%2 == 0:
                    action = simulation.get_random_action()
                else:
                    action = simulation.get_sinusoidal_action()
            
                obs, reward, done, trunc, info = simulation.step(action)
                
                # Render the environment
                simulation.render()
                
                # Check if the episode is done
                if done:
                    print(f"Task completed! Total rewards: {reward}")                    
                    print(colored("Environment reset complete.", "yellow"))
                    break

        except KeyboardInterrupt:
            print("Interrupted by user.")

    print("Done.")


    # STEP 10: Finish
    simulation.close()
    exit()

if __name__ == "__main__":
    main()
