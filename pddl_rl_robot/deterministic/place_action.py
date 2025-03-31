import numpy as np
from pddl_rl_robot.deterministic.deterministic_base_class import DeterministicBaseClass
from pddl_rl_robot.simulation.robot_controller import RobotController

class PlaceAction(DeterministicBaseClass):
    """
    Deterministic action to close the gripper.
    
    This class implements a simple policy that keeps the robot arm in place
    and closes the gripper to grasp an object.
    """
    
    def __init__(self, env):
        """
        Initialize the grasp action.
        
        Args:
            env: The environment the policy will interact with
        """
        super().__init__(env)
        self.controller = RobotController(env)
        
    def _get_action(self, observation):
        """
        Generate an action that keeps the arm in place and closes the gripper.
            
        Returns:
            action: An action vector with zeros for arm movement and gripper close command
        """
        return self.controller.set_gripper(-0.01) # Close gripper