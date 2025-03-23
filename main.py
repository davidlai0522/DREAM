from pddl_rl_robot.pddl.solver import PDDLSolver
from pddl_rl_robot.utils.config_loader import ConfigLoader
from pddl_rl_robot.simulation.simulator import RobotSimulator
import time
import numpy as np
from termcolor import colored
import os
import json
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
from pddl_rl_robot.rl.inference import RLModelInference
from pddl_rl_robot.deterministic.grasp_action import GraspAction
from pddl_rl_robot.deterministic.lift_action import LiftAction

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

DETERMINISTIC_ACTIONS = {
    "clasp-disk": GraspAction,
    "lift-disk-from-peg": LiftAction
}
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

class ActionType:
    """Enum class for action types."""
    DETERMINISTIC = 0
    REINFORCEMENT_LEARNING = 1
    NOT_RECOGNIZED = 2


def main():
    """Main function."""

    # STEP 0: Initialization
    NUM_STEP = 1000
    simulation = RobotSimulator(
        robot_type="Panda",
        env_name="two_peg_one_disk",
    )
    obs, _ = simulation.reset()

    print(f"The action space of Panda Robot is: {simulation.gym_env.action_space}")
    print(
        f"The observation space of Panda Robot is: {simulation.gym_env.observation_space}"
    )

    # ====================================== Load config ======================================
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config/project_config.yaml"
    )
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

    domain_path = os.path.join(
        PARENT_DIR,
        "config",
        config_loader.get_pddl_config()["domain_file"],
    )
    problem_path = os.path.join(
        PARENT_DIR,
        "config",
        config_loader.get_pddl_config()["problem_file"],
    )

    task_planner = PDDLSolver(domain_file=domain_path, problem_file=problem_path)

    print(colored("Solving the PDDL problem...", "cyan", attrs=["bold"]))
    plan = task_planner.solve()
    print(colored(f"Plan: {plan}", "green"))

    # ====================================== Exxecute the plan ======================================
    print(colored("-" * 50, "blue", attrs=["bold"]))
    print(colored("Executing the plan...", "cyan", attrs=["bold"]))

    # Plan: ['move(robot1, peg2, peg1)', 'reach-toward-disk-on-peg(robot1, disk1, peg1)', 'clasp-disk(robot1, disk1)', 'lift-disk-from-peg(robot1, disk1, peg1)', 'move(robot1, peg1, peg2)', 'place-disk-on-empty-peg(robot1, disk1, peg2)']

    action_type = ActionType.NOT_RECOGNIZED
    previous_action = None
    
    for idx, step in enumerate(plan):
        step = step.split()
        action = step[0].split("(")[0]
        parameters = step[1:]
        parameters = [p.replace(")", "") for p in parameters]
        
        # -------------------------------------------------------------------
        # Determine the action type
        # -------------------------------------------------------------------
        if (
            action in config_loader.get_deterministic_config()["actions"]
        ):
            action_type = ActionType.DETERMINISTIC
            deterministic_action = DETERMINISTIC_ACTIONS[action](env=simulation.gym_env)
            agent_config = config_loader.get_deterministic_config()[
                action
            ]
        elif (
            action in config_loader.get_reinforcement_learning_config()["agents"]
        ):
            action_type = ActionType.REINFORCEMENT_LEARNING
            agent_config = config_loader.get_reinforcement_learning_config()[
                action
            ]
            policy_inference = RLModelInference(
                env=simulation.gym_env,
                model_path=os.path.join(PARENT_DIR, agent_config["model"]),
                algorithm=agent_config["algorithm"],
            )
        else:
            print(
                colored(f"Action {action.split('(')[0]} is not a valid action.", "red")
            )
            continue
        
        
        print(
            colored(
                f"Executing action {idx+1}: \033[1m{action}\033[0m with parameters: {parameters} (Type: {'Deterministic' if action_type == ActionType.DETERMINISTIC else 'Reinforcement Learning' if action_type == ActionType.REINFORCEMENT_LEARNING else 'Unknown'})",
                "green",
            )
        )
        
        # -------------------------------------------------------------------
        # Execute the action
        # -------------------------------------------------------------------
        try:
            num_step = NUM_STEP
            if agent_config.get("max_step"):
                num_step = agent_config["max_step"]
            
            for _ in range(num_step):
                if action_type == ActionType.DETERMINISTIC:
                    deterministic_action.set_previous_action(previous_action)
                    action = deterministic_action.perform(obs)
                elif action_type == ActionType.REINFORCEMENT_LEARNING:
                    action, _ = policy_inference.model.predict(obs, deterministic=True)
                    
                obs, reward, done, _, _ = simulation.step(action)
                previous_action = action

                # Render the environment
                simulation.render()
                
                # Check if the episode is done
                if done:
                    print(colored("Environment reset complete.", "yellow"))
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        except Exception as e:
            print(colored(f"Error: {e}", "red"))
        print("=" * 50)

    print("Done.")

    # STEP 10: Finish
    simulation.close()
    exit()


if __name__ == "__main__":
    main()
