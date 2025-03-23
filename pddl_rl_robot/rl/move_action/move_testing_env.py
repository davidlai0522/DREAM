#!/usr/bin/env python3
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
import os
from pddl_rl_robot.rl.move_action.move_training_env import MoveTrainingEnv
import time

if __name__ == "__main__":
    env = MoveTrainingEnv(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
    env = GymWrapper(env)
    
    FINAL_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_nuts/ppo_panda_move_no_nuts_20250323_170533_100000.zip")
    
    model = PPO.load(f"{FINAL_MODEL_PATH}")

    print("Model Architecture:")
    print(model.policy)
    print("\nModel Hyperparameters:")
    print(f"learning_rate: {model.learning_rate}")  # Example of a hyperparameter
    print(f"n_steps: {model.n_steps}")  # Number of steps per update
    print(
        f"batch_size: {model.batch_size}"
    )  # Size of the batch for each training update
    print(f"gamma: {model.gamma}")  # Discount factor for rewards
    print(
        f"gae_lambda: {model.gae_lambda}"
    )  # GAE (Generalized Advantage Estimation) lambda

    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, _, info = env.step(action)
        env.render()
        time.sleep(0.2)
        if dones:
            print(f"Task completed! Total rewards: {rewards}")
            break
    env.close()
    print("Finish testing...")