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
class ReachTrainingEnv(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Override the reward function (please design it such that it favors reaching the goal)
    def reward(self, action=None):
        
        # """
        # Reward function focused specifically on reaching the nut.
        # """
        nut_id = self.sim.model.body_name2id('RoundNut_main')
        nut_pos = self.sim.data.body_xpos[nut_id]
        handle_pos = self.sim.data.get_site_xpos('RoundNut_handle_site')
        
        tip1_id = self.sim.model.body_name2id('gripper0_right_finger_joint1_tip')
        # tip1_id = self.sim.model.body_name2id( 'gripper0_right_leftfinger') #seems to be the rectangular thingy; failed
        tip1_pos = self.sim.data.body_xpos[tip1_id]
        
        tip2_id = self.sim.model.body_name2id('gripper0_right_finger_joint2_tip')
        # tip2_id = self.sim.model.body_name2id('gripper0_right_rightfinger') #seems to be the rectangular thingy; failed
        tip2_pos = self.sim.data.body_xpos[tip2_id]
        
        tip_pos = np.mean(np.array([tip1_pos, tip2_pos]), axis = 0)

        # ---- REACHING REWARD COMPONENT ----        
        # Base reward is inverse to distance (higher as gripper gets closer)
        # Using a scaled inverse distance function for smooth gradient
        dist_gripper_to_nut = np.linalg.norm(tip_pos - handle_pos)
        reach_reward = 1.0 / (1.0 + 5.0 * dist_gripper_to_nut)
        
        # Bonus rewards for getting very close
        if dist_gripper_to_nut < 0.08:
            reach_reward += 0.5  # Small bonus for getting close
            
        if dist_gripper_to_nut < 0.04:
            reach_reward += 1.0  # Larger bonus for getting very close
            
        if dist_gripper_to_nut < 0.02:
            reach_reward += 2.0  # Significant bonus for nearly touching

        # ---- ORIENTATION REWARD COMPONENT ----        
        # Add orientation reward (if gripper approaching from above)
        gripper_to_nut = handle_pos - tip_pos
        gripper_to_nut = gripper_to_nut / np.linalg.norm(gripper_to_nut)
        vertical_approach = np.dot(gripper_to_nut, np.array([0, 0, 1]))
        ori_reward = 0.5 * max(0, vertical_approach)  # Reward vertical approach

        # ---- OPEN GRIPPER REWARD COMPONENT ----
        # Reward keeping the gripper open - scales with openness
        fingertip_distance = np.linalg.norm(tip1_pos - tip2_pos)
        open_gripper_reward = 2.0 * fingertip_distance

        # ---- ACTION EFFICIENCY COMPONENT ----
        # Small penalty for large actions to encourage smooth motion
        action_magnitude = np.linalg.norm(action) if action is not None else 0
        action_penalty = 0.05 * action_magnitude

        # ---- FINAL REWARD CALCULATION ----
        # Final reward
        reward = reach_reward + ori_reward + open_gripper_reward - action_penalty
        return reward

if __name__ == "__main__":
    # Create environment instance
    env = ReachTrainingEnv(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
        has_renderer=False,  # Enable visualization
        has_offscreen_renderer=False,  # Disable offscreen rendering
        use_camera_obs=False,  # Don't use camera observations
        reward_shaping=False,  # Enable reward shaping
    )
    env = GymWrapper(env)

    # Configurations
    total_timesteps = 88888
    save_freq = 8888
    save_path = os.path.dirname(os.path.abspath(__file__))
    name_prefix = "ppo_panda_reach"

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
