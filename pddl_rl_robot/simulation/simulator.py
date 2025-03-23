"""
Robot Simulator for simulating robots using MuJoCo.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import mujoco
import mujoco.viewer
import threading
import time
import traceback
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
from robosuite.wrappers import GymWrapper

SIM_ENVS = {
    "two_peg_one_disk": TwoPegOneRoundNut
}

class RobotSimulator():
    """
    Simulator for robots using MuJoCo.
    """
    
    def __init__(self, robot_type: str = "Panda", env_name: str = "two_peg_one_disk", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the robot simulator.
        
        Args:
            robot_type: Type of robot
            env_name: Name of the environment
            config: Optional simulator configuration
        """
        self.robot_type = robot_type
        self.env_name = env_name
        self.config = config or {}

        self.gym_env = None
        self.env = None
        self.model = None
        self.viewer = None
        self.sim_thread = None
        self.stop_event = threading.Event()
        self.auto_render = False  # Disable auto-render to avoid segmentation fault
        self.is_initialized = False  # Flag to track if the environment is fully initialized
        self.has_rendered = False  # Flag to track if we've already rendered

        self.create_env()
    
    def create_env(self):
        """
        Creates the environment.
        """
        self.env = SIM_ENVS[self.env_name](
            robots=self.robot_type, 
            gripper_types="default",
            has_renderer=True,  # Enable visualization
            has_offscreen_renderer=False,  # Disable offscreen rendering
            use_camera_obs=False,  # Don't use camera observations
            reward_shaping=False,  # Enable reward shaping
            ignore_done=True,
            **self.config
        )
        self.model = self.env.model
        self.is_initialized = True

        self.gym_env = GymWrapper(self.env)

    def reset(self):
        """
        Reset the environment.
        """
        return self.gym_env.reset()

    def close(self):
        """
        Close the environment.
        """
        self.gym_env.close()

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
        
        Returns:
            The observation, reward, done, and info after taking the action
        """
        try:
            return self.gym_env.step(action)
        except ValueError as e:
            if "executing action in terminated episode" in str(e):
                # Environment has terminated, reset it and continue
                print("Environment terminated. Automatically resetting...")
                obs, _ = self.reset_env_keep_robot_position()
                # Return a dummy step result with done=False
                return obs, 0.0, False, False, {"auto_reset": True}
            else:
                # Re-raise other ValueError exceptions
                raise e

    def render(self):
        """
        Render the environment.
        """
        try:
            # Check if the viewer exists
            if hasattr(self.env, 'viewer') and self.env.viewer is None:
                # Initialize the viewer if it doesn't exist
                self.env._create_viewer()
                
            # Now render
            self.gym_env.render()
        except Exception as e:
            print(f"Error in render: {e}")
    
    def set_joint_pose(self, name, pose):
        """
        Set the pose of a joint.
        
        Args:
            name: The name of the joint
            pose: The pose to set
        """
        self.env.sim.data.set_joint_qpos(name, pose)

    def get_random_action(self):
        """
        Get a random action.
        
        Returns:
            A random action
        """
        return self.gym_env.action_space.sample()
        
    def get_sinusoidal_action(self, t=None):
        """
        Get a sinusoidal action.
        
        Args:
            t: Time parameter for the sinusoidal function
            
        Returns:
            A sinusoidal action
        """
        if t is None:
            t = time.time()
            
        action = np.zeros(self.gym_env.action_space.shape)
        for j in range(len(action)):
            action[j] = 0.1 * np.sin(t + j * np.pi / 4)
        return action
        
    def get_no_action(self):
        """
        Get a zero action (no movement).
        
        Returns:
            A zero action
        """
        return np.zeros(self.gym_env.action_space.shape)

    def get_current_reward(self):
        """
        Get the current reward.
        
        Returns:
            The current reward
        """
        return self.gym_env.reward_range[0]