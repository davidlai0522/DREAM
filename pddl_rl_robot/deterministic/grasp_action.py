import numpy as np
from pddl_rl_robot.deterministic.deterministic_base_class import DeterministicBaseClass


class GraspAction(DeterministicBaseClass):
    """
    Deterministic action to close the gripper.
    
    This class implements a simple policy that keeps the robot arm in place
    and closes the gripper to grasp an object.
    """
    
    def __init__(self, env, gripper_close_value=1.0):
        """
        Initialize the grasp action.
        
        Args:
            env: The environment the policy will interact with
            gripper_close_value: -1.0 to close the gripper, 1.0 to open the gripper
        """
        super().__init__(env)
        self.gripper_close_value = gripper_close_value
        self.grasp_steps = 0
        self.max_grasp_steps = 10  # Number of steps to complete the grasp
        self.object_grasped = False
        
    def _get_action(self, observation):
        """
        Generate an action that keeps the arm in place and closes the gripper.
            
        Returns:
            action: An action vector with zeros for arm movement and gripper close command
        """
        action = self.get_previous_action()
        if action is None:
            action = np.zeros(self.env.action_space.shape)
        action[-1] = self.gripper_close_value
        
        if self.grasp_steps > 0:
            self._update_object_position()
        self.grasp_steps += 1
        return action
            
    def _update_object_position(self):
        """
        Update the object's position to maintain its position relative to the gripper.
        This simulates a fixed connection between the gripper and the object.
        """
        # Get target body's position and orientation
        nut_id = self.env.sim.model.body_name2id('RoundNut_main')
        
        # Get the gripper finger IDs
        right_finger_id = self.env.sim.model.body_name2id('gripper0_right_finger_joint1_tip')
        left_finger_id = self.env.sim.model.body_name2id('gripper0_right_finger_joint2_tip')
        
        # Get current positions
        right_finger_pos = self.env.sim.data.body_xpos[right_finger_id]
        left_finger_pos = self.env.sim.data.body_xpos[left_finger_id]
        
        print(f"Right finger position: {right_finger_pos}")
        print(f"Left finger position: {left_finger_pos}")
        
        # Calculate the current gripper midpoint
        gripper_midpoint = (right_finger_pos + left_finger_pos) / 2
        
        # Calculate where the object should be based on the stored relative position
        target_pos = gripper_midpoint
        
        # Get the current object position
        current_pos = self.env.sim.data.body_xpos[nut_id]
        
        
        # Calculate the difference between current and target positions
        pos_diff = target_pos - current_pos
        print(f"Position difference: {pos_diff}")
        
        # Only update if the difference is significant
        if np.linalg.norm(pos_diff) > 0.001:  # Small threshold for precise tracking
            # Find the free object ID in qpos array
            # In MuJoCo, free objects typically have 7 DoF (3 for position, 4 for quaternion)
            # We need to find the correct index in qpos for the RoundNut
            
            # First, try to find the RoundNut's body ID and corresponding free joint
            for i in range(self.env.sim.model.njnt):
                joint_name = self.env.sim.model.joint_id2name(i)
                if joint_name and 'RoundNut' in joint_name and 'joint' in joint_name:
                    # Found the joint controlling the RoundNut
                    free_object_id = i
                    
                    # Get target orientation (keeping the current orientation of the nut)
                    target_quat = self.env.sim.data.body_xquat[nut_id]
                    
                    # Use the technique from the snippet to update position and orientation
                    # Set free object's position to match target position
                    print(f"Joint name: {joint_name}")
                    print(f"Free object ID: {free_object_id}")
                    # Print all joint names for debugging purposes
                    for j in range(self.env.sim.model.njnt):
                        print(f"Joint {j}: {self.env.sim.model.joint_id2name(j)}")
                    self.env.sim.data.qpos[free_object_id * 7 : free_object_id * 7 + 3] = target_pos
                    self.env.sim.data.qpos[free_object_id * 7 + 3 : free_object_id * 7 + 7] = target_quat
                    
                    # Forward the simulation to apply changes
                    self.env.sim.forward()
                    
                    print(f"Updated RoundNut position to {target_pos}")
                    break
            else:
                # If we can't find the joint directly, try a different approach
                # Find RoundNut in the qpos array using known patterns
                for obj_name in ['roundnut', 'round_nut', 'roundnut_joint', 'round_nut_joint']:
                    try:
                        # Try different possible naming conventions
                        obj_joint_id = self.env.sim.model.joint_name2id(obj_name)
                        free_object_id = obj_joint_id
                        
                        # Get the qpos address for this joint
                        qpos_addr = self.env.sim.model.jnt_qposadr[free_object_id]
                        
                        # Get current orientation
                        target_quat = self.env.sim.data.body_xquat[nut_id]
                        
                        # Update position and orientation
                        self.env.sim.data.qpos[qpos_addr : qpos_addr + 3] = target_pos
                        self.env.sim.data.qpos[qpos_addr + 3 : qpos_addr + 7] = target_quat
                        
                        # Forward the simulation
                        self.env.sim.forward()
                        
                        print(f"Updated RoundNut position using joint {obj_name}")
                        break
                    except:
                        continue
                else:
                    print("Could not find RoundNut joints. Trying direct body modification.")
                    # As a last resort, try to directly modify the body position
                    self.env.sim.data.body_xpos[nut_id] = target_pos
                    self.env.sim.forward()