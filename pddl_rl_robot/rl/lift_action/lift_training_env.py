from robosuite.wrappers import GymWrapper
import os
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
import numpy as np


# NOTE: this class is just a placeholder, please modify the reward based on some observations that are useful for reaching.
class LiftTrainingEnv(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_nut_pos = 0.0

    def _reset_internal(self):
        super()._reset_internal()
        self.robots[0].set_robot_joint_positions([0, -0.3, 0, -2.2, 0, 2.0, 0.8])
        
        
    # Override the reward function (please design it such that it favors reaching the goal)
    def reward(self, action=None):
        nut_pos = self.get_nut_pos()
        table_pos = self.get_object_position("table", "body")
        dist_gripper_to_nut = self.get_distance_from_gripper_to_nut_handle()
        gripper_ori = self.get_gripper_tip_ori(representation='euler')
        
        # Store initial orientation if not yet set
        if not hasattr(self, 'initial_gripper_ori'):
            self.initial_gripper_ori = self.get_gripper_tip_ori(representation='euler')
            self.initial_nut_pos = self.get_nut_pos()
            self.initial_table_height = table_pos[2]
            self.target_lift_height = 0.2  # Target 20 cm lift
            self.has_grasped = False
            
        # Calculate height from table
        height_from_table = nut_pos[2] - table_pos[2]
        
        # Determine if nut is grasped based on proximity and whether it's off the table
        is_grasped = dist_gripper_to_nut < 0.01 and height_from_table > 0.02
        if is_grasped and not self.has_grasped:
            self.has_grasped = True
            self.grasp_position = nut_pos.copy()
            
        # Base reward: proximity to nut (during reaching phase)
        reach_reward = 1 - np.tanh(5.0 * dist_gripper_to_nut)
        
        # Initialize specific rewards for the lifting phase
        lift_reward = 0.0
        orientation_reward = 0.0
        stability_reward = 0.0
        
        # If the nut is grasped, focus on lifting and orientation
        if self.has_grasped:
            # Reward for lifting height (normalized to target height)
            # Gradually increases as the nut gets closer to target height
            # and maxes out when it reaches the target height
            normalized_height = min(height_from_table / self.target_lift_height, 1.0)
            lift_reward = 3.0 * normalized_height
            
            # Extra reward for reaching target height
            if height_from_table >= self.target_lift_height:
                lift_reward += 1.0
                
            # Penalize horizontal movement during lifting
            horizontal_movement = np.sqrt((nut_pos[0] - self.grasp_position[0])**2 + 
                                          (nut_pos[1] - self.grasp_position[1])**2)
            stability_reward = -2.0 * horizontal_movement
            
            # Penalize orientation changes from initial grasp
            ori_diff = np.linalg.norm(gripper_ori - self.initial_gripper_ori)
            orientation_reward = -1.0 * ori_diff
            
            # Penalize if the nut isn't moving vertically
            if np.linalg.norm(nut_pos[2] - self.last_nut_pos[2]) < 1e-4 and height_from_table < self.target_lift_height:
                stability_reward -= 0.2
                
        else:
            # Penalties for pre-grasp phase
            if (nut_pos[2] > (table_pos[2] + 0.05) or 
                abs(nut_pos[0] - self.initial_nut_pos[0]) > 0.1 or 
                abs(nut_pos[1] - self.initial_nut_pos[1]) > 0.1):
                reach_reward -= 1.0
                
        # Update last nut position
        self.last_nut_pos = nut_pos.copy()
        
        # Combine rewards based on phase
        if self.has_grasped:
            # In lifting phase, prioritize lift, orientation, and stability
            reward = 0.2 * reach_reward + 0.5 * lift_reward + 0.15 * orientation_reward + 0.15 * stability_reward
        else:
            # In reaching phase, prioritize reaching the nut
            reward = reach_reward
        
        return reward


if __name__ == "__main__":
    # Create environment instance
    env = LiftTrainingEnv(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
        # has_renderer=False,  # Enable visualization
        has_renderer=True,  # Enable visualization
        has_offscreen_renderer=False,  # Disable offscreen rendering
        use_camera_obs=False,  # Don't use camera observations
        reward_shaping=False,  # Enable reward shaping
    )
    env = GymWrapper(env)

    # Configurations
    total_timesteps = 88888
    save_freq = 8888
    save_path = os.path.dirname(os.path.abspath(__file__))
    name_prefix = "ppo_panda_lift"

    # Initialize the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=name_prefix,
        save_replay_buffer=True,
    )

    # Define the model
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"{os.path.dirname(os.path.abspath(__file__))}/{name_prefix}_{current_time}_{str(total_timesteps)}")
    env.close()
    print(f"Training completed. Model saved to {os.path.dirname(os.path.abspath(__file__))}/{name_prefix}_{current_time}_{str(total_timesteps)}")

    exit()