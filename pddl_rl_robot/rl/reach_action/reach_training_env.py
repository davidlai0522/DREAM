from robosuite.wrappers import GymWrapper
import os
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime

# NOTE: this class is just a placeholder, please modify the reward based on some observations that are useful for reaching.
class ReachTrainingEnv(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.horizon =128 # cannot be too short, will cause instability. 
        
    # Override the reward function (please design it such that it favors reaching the goal)
    def reward(self, action=None):
        
        # Get positions of nut handle and gripper tips
        nut_handle_name = self.nuts[0].important_sites['handle']
        nut_handle_id = self.sim.model.site_name2id(nut_handle_name)
        nut_handle_pos = self.sim.data.site_xpos[nut_handle_id]
        nut_pos = np.mean([nut_handle_pos], axis=0)
        
        tip1_id = self.sim.model.body_name2id('gripper0_right_finger_joint1_tip')
        tip1_pos = self.sim.data.body_xpos[tip1_id]
        tip2_id = self.sim.model.body_name2id('gripper0_right_finger_joint2_tip')
        tip2_pos = self.sim.data.body_xpos[tip2_id]
        tip_center = np.mean(np.array([tip1_pos, tip2_pos]), axis=0)
        
        # Core geometric calculations
        gripper_to_nut = nut_pos - tip_center
        dist_gripper_to_nut = np.linalg.norm(gripper_to_nut)
        gripper_axis = tip2_pos[:2] - tip1_pos[:2]
        gripper_width = np.linalg.norm(gripper_axis)
        gripper_axis_norm = gripper_axis / (gripper_width + 1e-6)
        
        # ===== NORMALIZED REWARD COMPONENTS =====
        
        # 1. Reach reward - base distance component
        reach_reward = 20 / (1 + 18 * dist_gripper_to_nut)
        if dist_gripper_to_nut <= 0.15: reach_reward += 0.2
        if dist_gripper_to_nut <= 0.10: reach_reward += 0.8
        if dist_gripper_to_nut <= 0.05: reach_reward += 1.8
        if dist_gripper_to_nut <= 0.02: reach_reward += 2.5
            
        # 3. Alignment reward - perpendicular distance to gripper axis
        dist_tip1_to_nut = np.linalg.norm(nut_pos - tip1_pos)
        dist_tip2_to_nut = np.linalg.norm(nut_pos - tip2_pos)
        alignment_diff = abs((dist_tip1_to_nut / dist_tip2_to_nut) - 1)
        alignment_reward = 1 * np.exp(10 * -alignment_diff)
        
        # 4. Gripper orientation reward - keep fingers level
        
        height_diff = abs(tip1_pos[2] - tip2_pos[2])
        parallel_reward = 1 * np.exp(10 * -height_diff)
        width_reward = 1 * np.exp(10 * gripper_width)
        
        # Add vertical reward (if gripper approaching from above)
        gripper_to_nut = nut_pos - tip_center
        gripper_to_nut = gripper_to_nut / np.linalg.norm(gripper_to_nut)
        vertical_approach = np.dot(gripper_to_nut, np.array([0, 0, 1]))
        vertical_reward = 0.5 * max(0, vertical_approach)  # Reward vertical approach
        
        # 6. Action efficiency - encourage smooth motions
        if hasattr(self, 'previous_action'):
            action_diff = np.linalg.norm(action - self.previous_action)
            action_penalty = 0.2 * action_diff  # Penalize large changes
        else: action_penalty = 0
        self.previous_action = action
        
        total_reward = (reach_reward +
                        alignment_reward + 
                        vertical_reward + 
                        parallel_reward + 
                        width_reward -
                        action_penalty
                       )

        if ((dist_gripper_to_nut <= 0.01) & 
            (alignment_diff <= 0.08) & 
            (height_diff <= 0.03) & 
            (gripper_width >= 0.07)
           ):
            total_reward = total_reward + 88

        if (gripper_width < 0.07):
            total_reward =  0
         
        self.dist_gripper_to_nut = dist_gripper_to_nut
        self.alignment_diff = alignment_diff
        self.height_diff = height_diff
        self.gripper_width = gripper_width
        
        return total_reward

    def _check_success(self):
        # to call this, use env.unwrapped._check_success()

        # Get positions of nut handle and gripper tips
        nut_handle_name = self.nuts[0].important_sites['handle']
        nut_handle_id = self.sim.model.site_name2id(nut_handle_name)
        nut_handle_pos = self.sim.data.site_xpos[nut_handle_id]
        nut_pos = np.mean([nut_handle_pos], axis=0)
        
        tip1_id = self.sim.model.body_name2id('gripper0_right_finger_joint1_tip')
        tip1_pos = self.sim.data.body_xpos[tip1_id]
        tip2_id = self.sim.model.body_name2id('gripper0_right_finger_joint2_tip')
        tip2_pos = self.sim.data.body_xpos[tip2_id]
        tip_center = np.mean(np.array([tip1_pos, tip2_pos]), axis=0)
        
        # Core geometric calculations
        gripper_to_nut = nut_pos - tip_center
        dist_gripper_to_nut = np.linalg.norm(gripper_to_nut)
        gripper_axis = tip2_pos[:2] - tip1_pos[:2]
        gripper_width = np.linalg.norm(gripper_axis)

        dist_tip1_to_nut = np.linalg.norm(nut_pos - tip1_pos)
        dist_tip2_to_nut = np.linalg.norm(nut_pos - tip2_pos)
        alignment_diff = abs((dist_tip1_to_nut / dist_tip2_to_nut) - 1)

        height_diff = abs(tip1_pos[2] - tip2_pos[2])

        if ((dist_gripper_to_nut <= 0.01) & (alignment_diff <= 1.5) & (height_diff <= 0.1) & (gripper_width >= 0.07)):
            return True
            
        return False

if __name__ == "__main__":
    # Create environment instance
    env = ReachTrainingEnv(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
        has_renderer=False,  # Enable visualization
        has_offscreen_renderer=True,  # Disable offscreen rendering
        use_camera_obs=False,  # Don't use camera observations
        reward_shaping=False,  # Enable reward shaping
    )
    env = GymWrapper(env)

    # Configurations
    total_timesteps = 88888
    save_freq = 8888
    save_path = os.path.dirname(os.path.abspath(__file__))
    name_prefix = "ppo_panda_reach_20250323"

    # Initialize the checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_path,
        name_prefix=name_prefix,
        save_replay_buffer=True,
    )

    # Define the model
    model = PPO("MlpPolicy", env, verbose=0,
                seed= 8888, learning_rate=0.0003, batch_size = 64, 
                n_epochs = 10, n_steps = 2048,) #do not decrease time step and batch size together.

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"{os.path.dirname(os.path.abspath(__file__))}/{name_prefix}_{current_time}_{str(total_timesteps)}")
    env.close()

    exit()
