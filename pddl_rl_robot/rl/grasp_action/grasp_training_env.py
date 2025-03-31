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
class GraspTrainingEnv(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _reset_internal(self):
        super()._reset_internal()
        
        self.robots[0].set_robot_joint_positions([0, -0.3, 0, -2.2, 0, 2.0, 0.8])
        

    # Override the reward function (please design it such that it favors grasping the goal)
    def reward(self, action=None):
        # Get key positions
        nut_pos = self.get_nut_pos()
        handle_pos = self.sim.data.get_site_xpos('RoundNut_handle_site')
        gripper_pos = self.get_gripper_tip_pos()
        gripper_ori = self.get_gripper_tip_ori(representation='euler')
        dist_gripper_to_nut = self.get_distance_from_gripper_to_nut_handle()
        is_grasped = self.check_grasp()

        # Store initial orientation if not yet set
        if not hasattr(self, 'initial_gripper_ori'):
            self.initial_gripper_ori = self.get_gripper_tip_ori(representation='euler')
            self.has_grasped = False
            self.grasp_attempts = 0
            self.max_grasp_attempts = 5

        # Proximity reward (higher as gripper gets closer to nut handle)
        proximity_reward = 1.0 / (1.0 + 5.0 * dist_gripper_to_nut)

        # Orientation reward (maintain initial orientation)
        ori_diff = np.linalg.norm(gripper_ori - self.initial_gripper_ori)
        orientation_reward = 1.0 / (1.0 + 2.0 * ori_diff)  # Higher when orientation is maintained

        # Grasp reward
        grasp_reward = 0.0
        if is_grasped:
            grasp_reward = 5.0
            if not self.has_grasped:
                grasp_reward += 10.0  # Bonus for first successful grasp
                self.has_grasped = True
        elif action is not None and abs(action[-1]) > 0.5:  # Attempted grasp
            self.grasp_attempts += 1
            if self.grasp_attempts > self.max_grasp_attempts:
                grasp_reward -= 0.5  # Penalty for excessive failed grasp attempts

        # Combine rewards with appropriate weights
        total_reward = (
            2.0 * proximity_reward +
            1.0 * orientation_reward +
            3.0 * grasp_reward
        )

        return total_reward


if __name__ == "__main__":
    # Create environment instance
    env = GraspTrainingEnv(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
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
    name_prefix = "ppo_panda_grasp"

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
