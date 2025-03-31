import numpy as np
from pddl_rl_robot.deterministic.deterministic_base_class import DeterministicBaseClass
from pddl_rl_robot.simulation.robot_controller import RobotController


class LiftAction(DeterministicBaseClass):
    """
    Deterministic action to lift an object after it has been grasped.
    
    This class implements a simple policy that moves the robot arm upward
    while keeping the gripper closed to lift a grasped object.
    """
    
    def __init__(self, env, gripper_close_value=1.0, lift_height=0.2):
        super().__init__(env)
        self.controller = RobotController(env)
        self.gripper_close_value = gripper_close_value
        self.lift_height = lift_height
        self.lift_steps = 0
        self.max_lift_steps = 50  # Number of steps to complete the lift action
        self.object_grasped = False
        self.controller = RobotController(env)
        
    def _get_action(self, observation):
        """
        Generate an action that moves the robot arm upward while keeping the gripper closed.
            
        Returns:
            action: An action vector with joint movements for lifting and closed gripper
        """
        # Get current position and only change the z coordinate
        current_position = self.controller.get_eef_position()
        target_position = np.array([current_position[0], current_position[1], 1.0])
        self.controller.attach_object_to_eef("RoundNut", offset=np.array([0.05, 0.0, 0.02]))
        return self.controller.move_to_position(target_position)