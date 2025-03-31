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
class PlaceTrainingEnv(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reward(self, action=None):
        # Find gripper location, may change to nut position or nut handle 
        #nut_id = self.sim.model.body_name2id('RoundNut_main')
        #nut_pos = self.sim.data.body_xpos[nut_id] 
        gripper_pos = self.sim.data.get_site_xpos('gripper0_right_ee_z') #gripper position
        #tip1_id = self.sim.model.body_name2id('gripper0_right_finger_joint1_tip')
        #tip1_pos = self.sim.data.body_xpos[tip1_id]
        #tip2_id = self.sim.model.body_name2id('gripper0_right_finger_joint2_tip')
        #tip2_pos = self.sim.data.body_xpos[tip2_id]
        #tip_pos = np.mean(np.array([tip1_pos, tip2_pos]), axis = 0)
        
        table_pos = np.array(self.sim.data.body_xpos[self.table_body_id])
        z_target = 0.8 #table_pos[2]+0.125  #target height for lift is 0.95, target height for place is 0.8
        dists = abs(z_target-gripper_pos[2]) #distance from target height
        reward = 1.0 / (1.0 + 10 * dists)
        # print(z_target, gripper_pos[2], reward)
        return reward

     # override post action to check if the episode is done (reward near target)
    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method
        """
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        if reward > 0.99:
            print("Target reached")
            self.done = True
        return reward, self.done, {}

    # override the step function to restrict the action space
    def step(self, action):
        z_action = action[2]
        action = np.zeros(7) 
        action[2] = z_action    #restrict action to z axis
        action[6] = 1           #gripper closed
        #Clip the action to the restricted bounds
        action = np.clip(action, self.action_spec[0], self.action_spec[1])
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):
            if self.lite_physics:
                self.sim.step1()
            else:
                self.sim.forward()
            self._pre_action(action, policy_step)
            if self.lite_physics:
                self.sim.step2()
            else:
                self.sim.step()
            self._update_observables()
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)

        if self.viewer is not None and self.renderer != "mujoco":
            self.viewer.update()
        elif self.has_renderer and self.renderer == "mjviewer" and self.viewer is None:
            # need to launch again after it was destroyed
            self.initialize_renderer()
            # so that mujoco viewer renders
            self.viewer.update()

        observations = self.viewer._get_observations() if self.viewer_get_obs else self._get_observations()
        return observations, reward, done, info

if __name__ == "__main__":
    # Create environment instance
    env = PlaceTrainingEnv(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
        has_renderer=True,  # Enable visualization
        has_offscreen_renderer=False,  # Disable offscreen rendering
        use_camera_obs=False,  # Don't use camera observations
        reward_shaping=False,  # Enable reward shaping
    )
    env = GymWrapper(env)

    # Configurations
    total_timesteps = 10000
    save_freq = 1000
    save_path = os.path.dirname(os.path.abspath(__file__))
    name_prefix = "ppo_panda_place"

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
    model.load(f"{os.path.dirname(os.path.abspath(__file__))}/place_train_model")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #model.save(f"{os.path.dirname(os.path.abspath(__file__))}/place_train_model") #{name_prefix}_{current_time}_{str(total_timesteps)}")

    env.close()

    exit()
