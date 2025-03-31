from robosuite.wrappers import GymWrapper
import os
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime

# NOTE: this class is just a placeholder, please modify the reward based on some observations that are useful for reaching.
class PickTrainingEnv(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.horizon =128 # cannot be too short, will cause instability. 

    # Override the reward function (please design it such that it favors reaching the goal)
    def reward(self, action=None):
        
        tip1_id = self.sim.model.body_name2id('gripper0_right_finger_joint1_tip')
        tip1_pos = self.sim.data.body_xpos[tip1_id]
        tip2_id = self.sim.model.body_name2id('gripper0_right_finger_joint2_tip')
        tip2_pos = self.sim.data.body_xpos[tip2_id]
        
        gripper_axis = tip2_pos[:2] - tip1_pos[:2]
        gripper_width = np.linalg.norm(gripper_axis)

        width_reduction = 9
        if hasattr(self, 'gripper_width'):
            width_reduction = self.gripper_width - gripper_width
        self.width_reduction = width_reduction
        self.gripper_width = gripper_width
        
        width_reward = 3 - np.exp(10 *gripper_width)
        if self.width_reduction <= 0.002: 
            width_reward = 0
        
        total_reward = width_reward
        
        return total_reward
        
    def _check_success(self):

        # to call this, use env.unwrapped._check_success()
        if self.width_reduction <= 0.002:
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
    total_timesteps = 8888
    save_freq = 8888
    save_path = os.path.dirname(os.path.abspath(__file__))
    name_prefix = "ppo_panda_pick_20250323"

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