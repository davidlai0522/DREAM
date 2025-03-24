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



class MoveTrainingEnv_no_nut(TwoPegOneRoundNut):
    """
    Modified version of task - place one round nut with two pegs in the environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_peg_name = 'peg1'
        self.target_peg_id = self.sim.model.body_name2id(self.target_peg_name)

        self.original_peg_name = 'peg2'
        self.original_peg_id = self.sim.model.body_name2id(self.original_peg_name)
        
        self.height_threshold = [0.07, 0.1]
        self.nut_angle_threshold = np.pi/10

        self.nut_to_target_peg_xy_dist_threshold = 0.01
    
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
        print("reset internal")

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

    def _gripper_angle(self):
        """
        When the gripper is pointing downward, angle = pi
        """
        id = self.sim.model.body_name2id('gripper0_right_finger_joint1_tip')
        quat = T.convert_quat(self.sim.data.body_xquat[id], to="xyzw")
        mat = T.quat2mat(quat)
        z_unit = [0, 0, 1]
        z_rotated = np.matmul(mat, z_unit)
        z_rotated = z_rotated / np.linalg.norm(z_rotated)
        angle = np.arccos(np.dot(z_unit, z_rotated))

        return angle
    
    def _pre_action(self, action, policy_step=False):
        """
        Overrides the superclass method to control the robot(s) within this enviornment using their respective
        controllers using the passed actions and gripper control.

        Args:
            action (np.array): The control to apply to the robot(s). Note that this should be a flat 1D array that
                encompasses all actions to be distributed to each robot if there are multiple. For each section of the
                action space assigned to a single robot, the first @self.robots[i].controller.control_dim dimensions
                should be the desired controller actions and if the robot has a gripper, the next
                @self.robots[i].gripper.dof dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """
        action[6] = 1

        # Verify that the action is the correct dimension
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))

        # Update robot joints based on controller actions
        cutoff = 0
        for idx, robot in enumerate(self.robots):
            robot_action = action[cutoff:cutoff+robot.action_dim]
            robot.control(robot_action, policy_step=policy_step)
            cutoff += robot.action_dim

    # Override the reward function (please design it such that it favors reaching the goal)
    def reward(self, action=None):

        _, _, _, target_peg_xpos = self._get_task_info()

        reward = 0

        if self._check_success():
            reward = 10
            print(f"SUCCESS: Reward = {reward}")

        gripper_to_target_vec = self._eef0_xpos[:2] - target_peg_xpos[:2]

        # distance
        dist_to_target_peg = np.linalg.norm(gripper_to_target_vec)
        reaching_reward = 4 * (1 - np.tanh(5.0 * dist_to_target_peg))


        # height 
        height_reward = 0
        curr_height = self._eef0_xpos[2]
        lower_threshold, higher_threshold = target_peg_xpos[2] + self.height_threshold[0], target_peg_xpos[2] + self.height_threshold[1]

        if curr_height >= lower_threshold and curr_height <= higher_threshold:
            # higher than lower-threshold, prevent from getting over the higher threshold
            height_reward = 1 + np.tanh(10 * abs((lower_threshold + higher_threshold)/2 - curr_height))
        elif curr_height < lower_threshold:
            height_reward = 1 - np.tanh(10 * (lower_threshold - curr_height))
        else:
            height_reward = 1 - np.tanh(10 * (curr_height - higher_threshold))


    
        # orientation
        gripper_angle = self._gripper_angle()

        ori_reward = 2.0 * np.tanh(gripper_angle - 0.8 * np.pi)

        # Small penalty for large actions to encourage smooth motion
        if hasattr(self, 'previous_action'):
            action_diff = np.linalg.norm(action[:-1] - self.previous_action[:-1])
            action_penalty = 1.25 * np.tanh(2.5 * action_diff)  # Penalize large changes
        else: 
            action_penalty = 0
        self.previous_action = action

        reward = reaching_reward + height_reward + ori_reward - action_penalty

        return reward
    
    def _check_success(self):
        _, _, _, target_peg_xpos = self._get_task_info()

        xy_dist_to_peg = np.linalg.norm(self._eef0_xpos[:2] - target_peg_xpos[:2])

        gripper_angle = self._gripper_angle()

        if xy_dist_to_peg <= self.nut_to_target_peg_xy_dist_threshold \
            and self._eef0_xpos[2] >= target_peg_xpos[2] + self.height_threshold[0] \
            and self._eef0_xpos[2] <= target_peg_xpos[2] + self.height_threshold[1] \
                and (np.pi - gripper_angle) < np.pi/10:
            return True
        
        return False

if __name__ == "__main__":
    print("??")
    # Create environment instance
    env = MoveTrainingEnv_no_nut(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
        has_renderer=False,  # Enable visualization
        has_offscreen_renderer=False,  # Disable offscreen rendering
        use_camera_obs=False,  # Don't use camera observations
        reward_shaping=False,  # Enable reward shaping
    )
    env = GymWrapper(env)

    # Configurations
    total_timesteps = 100000
    save_freq = 5000
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_nuts")
    name_prefix = "ppo_panda_move_no_nuts"

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
