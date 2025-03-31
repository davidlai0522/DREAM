#!/usr/bin/env python3
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from pddl_rl_robot.rl.lift_action.lift_training_env import LiftTrainingEnv
import os
import numpy as np

if __name__ == "__main__":
    # Create environment instance
    env = LiftTrainingEnv(
        robots="Panda",
        gripper_types="default",
        has_renderer=True,  # Enable visualization for testing
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=False,
    )
    env = GymWrapper(env)
    
    # Load the trained model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, "ppo_panda_lift_20250323_004539_88888.zip")
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Test the model
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()
        if done:
            print(f"Task completed! Total rewards: {total_reward}")
            break
    env.close()
