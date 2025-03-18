from datetime import datetime
import os

import numpy as np
from robosuite.wrappers import GymWrapper
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
import robosuite.utils.transform_utils as T
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback


from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut



class MoveTrainingEnv(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_peg_name = 'peg1'
        self.target_peg_id = self.sim.model.body_name2id(self.target_peg_name)
        self.height_threshold = 0.1
        self.nut_angle_threshold = np.pi/10
        self.nut_to_target_peg_xy_dist_threshold = 0.05 / 2
    
    @property
    def _eef0_xpos(self):
        """
        Grab the position of Robot 0's right end effector.

        Returns:
            np.array: (x,y,z) position of EEF0
        """

        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])

    def _reset_internal(self):
        """
        Resets the environment by setting the robot's hand to hold the nut.
        """
        ManipulationEnv._reset_internal(self)

        for _ in range(100):
            # Iteratively update the nut handle's position based on the position of the gripper
            handle_to_nut_vec = self.get_nut_pos() - self.get_object_position(target = self.nuts[0].important_sites['handle'], target_type="site")
            self.sim.data.set_joint_qpos(
                self.nuts[0].joints[0], np.concatenate([self._eef0_xpos + handle_to_nut_vec, self.get_nut_ori()])
            )
            
            # Close gripper (action = 1) and prevent arm from moving
            gripper_ac = [1] * self.robots[0].gripper["right"].dof
            gripper_ac = self.robots[0].gripper["right"].format_action(gripper_ac)
            self.robots[0].part_controllers["right_gripper"].set_goal(gripper_ac)

            # Take forward step
            self.sim.step()
    

    def _nut_quat(self):
        """
        Grab the orientation of the nut body.

        Returns:
            np.array: (x,y,z,w) quaternion of the nut body
        """
        nut_id = self.obj_body_id[self.nuts[0].name]
        return T.convert_quat(self.sim.data.body_xquat[nut_id], to="xyzw")


    def _nut_angle(self):
        """
        Calculate the angle of nut with the ground. Returns 0 when it is lying flat, pi/2 when it is standing

        Returns:
            float: angle in radians
        """
        mat = T.quat2mat(self._nut_quat())
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        return np.arccos(np.dot(z_unit, z_rotated))
    
    def _get_task_info(self):
        """
        Helper function that grabs the current relevant locations of objects of interest within the environment

        Returns:
            4-tuple:

                - (bool) True if the arm is grasping the nut
                - (float) Height of table
                - (ndarray) xyz position of nut
                - (ndarray) xyz position of target peg

        """
        
        nut_id = self.obj_body_id[self.nuts[0].name]

        # check grasping status
        grasped = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.nuts[0].contact_geoms) #object_geoms=self.nuts[0].contact_geoms
        
        # table height
        table_height = self.sim.data.body_xpos[self.sim.model.body_name2id('table')][2]

        # xyz position of nut and target peg
        nut_xpos = np.array(self.sim.data.body_xpos[nut_id])[:3]
        target_peg_xpos = np.array(self.sim.data.body_xpos[self.target_peg_id])[:3]
        
        return grasped, table_height, nut_xpos, target_peg_xpos
    
    
    # Override the reward function (please design it such that it favors reaching the goal)
    def reward(self, action=None):
        """
        Reward function for the task of moving the nut to the space above the target peg

        Shaped rewards:
            - Grasping: in {0, 0.5}, nonzero if Arm 0 is gripping the nut.
            - Lifting: in {0, [1.0, 1.5]}, nonzero if Arm 0 lifts the handle above the height of both pegs
            - Delivering: in {0, [1.0, 1.25]}, nonzero only if Arm0 is grasping the nut, and is
                proportional to the distance between the handle and Arm 1
            - Hovering: (6) in {0, 2.0}, nonzero when only Arm 1 is gripping the nut and has the nut
                lifted above the target peg, i.e. the (x,y) positions are close

        """

        grasped, table_height, nut_xpos, target_peg_xpos = self._get_task_info()
        
        nut_height = nut_xpos[2]
        target_peg_height = target_peg_xpos[2]

        if nut_height - target_peg_height > self.height_threshold:
            reward = 1.0
            # Add in up to 0.25 based on distance between handle and gripper
            dist = np.linalg.norm(nut_xpos[:2] - target_peg_xpos[:2])
            reaching_reward = 0.25 * (1 - np.tanh(1.0 * dist))
            reward += reaching_reward

            # Add in up to 0.25 based on the angle of the nut
            nut_angle_reward = self._nut_angle_reward(self._nut_angle())
            reward += nut_angle_reward

        # the nut is not raised high enough, or not even being grasped
        else:
            # Split cases depending on whether arm0 is currently grasping the handle or not
            if grasped:
                reward = 0.5
            else:
                # encourage arm0 to reach for the handle
                dist = np.linalg.norm(self.get_distance_from_gripper_to_nut_handle())
                reaching_reward = 0.25 * (1 - np.tanh(1.0 * dist))
                reward = reaching_reward
        
        
        return reward
    
    def _nut_angle_reward(self, angle):
        '''
        Encourages the nut to be held with its hollow part upward/downward facing.
            :param angle:  float value in range [0, pi]
        '''
        if angle >= np.pi/2:
            return 0.25*(1 - np.tanh(np.pi - angle))
        else:
            return 0.25*(1 - np.tanh(angle - 0))
     
    def _check_success(self):
        """
        Always returns False to prevent the environment from terminating.
        """
        _, _, nut_xpos, target_peg_xpos = self._get_task_info()

        xy_dist_to_peg = np.linalg.norm(nut_xpos[:2] - target_peg_xpos[:2])

        nut_angle = self._nut_angle()
        if nut_angle >= np.pi/2:
            nut_angle = np.pi - nut_angle

        if nut_angle < self.nut_angle_threshold \
            and nut_xpos[2] > target_peg_xpos[2] \
                and xy_dist_to_peg < self.nut_to_target_peg_xy_dist_threshold:
            return True
        
        return False

if __name__ == "__main__":
    # Create environment instance
    env = MoveTrainingEnv(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
        has_renderer=True,  # Enable visualization
        has_offscreen_renderer=False,  # Disable offscreen rendering
        use_camera_obs=False,  # Don't use camera observations
        reward_shaping=False,  # Enable reward shaping
    )
    env = GymWrapper(env)

    # Configurations
    total_timesteps = 100000
    save_freq = 5000
    save_path = os.path.dirname(os.path.abspath(__file__))
    name_prefix = "ppo_panda_move"

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

    exit()
