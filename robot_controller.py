import numpy as np
import time

class RobotController:
    """
    A class to simplify the control of a robot in the robosuite environment.
    This controller provides methods for direct joint control, end-effector positioning,
    and object manipulation.
    """
    
    def __init__(self, env):
        """
        Initialize the robot controller.
        
        Args:
            env: The robosuite environment instance
        """
        self.env = env
        self.robot = env.robots[0]  # Assume first robot
        self.sim = env.sim
        
        # Initialize robot joint indices and addresses
        self.robot_joint_indices = []
        self.robot_qpos_addr = []
        self._initialize_robot_joints()
        
        # Initialize end-effector site
        self.eef_site_id = self._find_eef_site()
        
        # Initialize object information
        self.object_ids = {}
        self.object_joint_ids = {}
        self._initialize_object_info()
        
        # Get action dimension
        self.action_dim = env.action_spec[0].shape[0]
        
        # Find gripper indices
        self.gripper_indices = self._find_gripper_indices()
    
    def _initialize_robot_joints(self):
        """Find and store the robot's joint indices and qpos addresses."""
        for i in range(self.sim.model.njnt):
            joint_name = self.sim.model.joint_id2name(i)
            if joint_name and "robot0" in joint_name:
                self.robot_joint_indices.append(i)
                addr = self.sim.model.jnt_qposadr[i]
                self.robot_qpos_addr.append(addr)
    
    def _find_eef_site(self):
        """Find the end-effector site ID."""
        for i in range(self.sim.model.nsite):
            site_name = self.sim.model.site_id2name(i)
            if site_name and "grip_site" in site_name:
                return i
        return None
    
    def _initialize_object_info(self):
        """Find and store object body and joint IDs."""
        # Find object bodies
        for i in range(self.sim.model.nbody):
            body_name = self.sim.model.body_id2name(i)
            if body_name:
                if "RoundNut" in body_name:
                    self.object_ids["RoundNut"] = i
        
        # Find object joints
        for i in range(self.sim.model.njnt):
            joint_name = self.sim.model.joint_id2name(i)
            if joint_name:
                if "RoundNut" in joint_name:
                    self.object_joint_ids["RoundNut"] = i
    
    def _find_gripper_indices(self):
        """Find and store gripper joint indices."""
        gripper_indices = []
        for i in range(self.sim.model.njnt):
            joint_name = self.sim.model.joint_id2name(i)
            if joint_name and "robot0" in joint_name and "finger" in joint_name:
                gripper_indices.append(i)
        return gripper_indices
    
    def get_joint_positions(self):
        """
        Get the current robot joint positions.
        
        Returns:
            numpy.ndarray: Array of joint positions
        """
        qpos = self.sim.data.qpos.copy()
        joint_positions = []
        for addr in self.robot_qpos_addr:
            joint_positions.append(qpos[addr])
        return np.array(joint_positions)
    
    def set_joint_positions(self, positions):
        """
        Set the robot joint positions directly.
        
        Args:
            positions (numpy.ndarray): Array of joint positions (must match the number of robot joints)
        """
        if len(positions) != len(self.robot_qpos_addr):
            raise ValueError(f"Expected {len(self.robot_qpos_addr)} joint positions, got {len(positions)}")
        
        qpos = self.sim.data.qpos.copy()
        for i, addr in enumerate(self.robot_qpos_addr):
            qpos[addr] = positions[i]
        
        self.sim.data.qpos[:] = qpos
        self.sim.forward()
    
    def get_eef_position(self):
        """
        Get the end-effector position.
        
        Returns:
            numpy.ndarray: 3D position of the end-effector
        """
        if self.eef_site_id is not None:
            return self.sim.data.site_xpos[self.eef_site_id].copy()
        return None
    
    def get_eef_orientation(self):
        """
        Get the end-effector orientation.
        
        Returns:
            numpy.ndarray: Quaternion orientation of the end-effector (w, x, y, z)
        """
        if self.eef_site_id is not None:
            # Get the rotation matrix
            rot_matrix = self.sim.data.site_xmat[self.eef_site_id].reshape(3, 3)
            
            # Convert rotation matrix to quaternion
            # This is a simplified conversion - may need adjustment based on coordinate system
            trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
            
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
                y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
                z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
            elif rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2])
                w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2])
                w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1])
                w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                z = 0.25 * s
            
            return np.array([w, x, y, z])
        return None
    
    def set_object_position(self, object_name, position):
        """
        Set the position of an object directly.
        
        Args:
            object_name (str): Name of the object (e.g., "RoundNut")
            position (numpy.ndarray): 3D position to set
        """
        if object_name not in self.object_joint_ids:
            raise ValueError(f"Object {object_name} not found")
        
        joint_id = self.object_joint_ids[object_name]
        qpos_addr = self.sim.model.jnt_qposadr[joint_id]
        
        # Check if it's a free joint (7 DoF) or another type
        joint_type = self.sim.model.jnt_type[joint_id]
        
        qpos = self.sim.data.qpos.copy()
        if joint_type == 0:  # Free joint
            # Position (xyz)
            qpos[qpos_addr:qpos_addr+3] = position
            # Keep the current orientation (qpos[qpos_addr+3:qpos_addr+7])
        else:
            # For other joint types, just set the position directly
            qpos[qpos_addr] = position[0]
        
        self.sim.data.qpos[:] = qpos
        self.sim.forward()
    
    def get_object_position(self, object_name):
        """
        Get the position of an object.
        
        Args:
            object_name (str): Name of the object (e.g., "RoundNut")
            
        Returns:
            numpy.ndarray: 3D position of the object
        """
        if object_name not in self.object_ids:
            raise ValueError(f"Object {object_name} not found")
        
        body_id = self.object_ids[object_name]
        return self.sim.data.body_xpos[body_id].copy()
    
    def attach_object_to_eef(self, object_name, offset=np.array([0.0, 0.0, 0.0])):
        """
        Attach an object to the end-effector with a specified offset.
        
        Args:
            object_name (str): Name of the object to attach
            offset (numpy.ndarray): 3D offset from the end-effector position
        """
        eef_pos = self.get_eef_position()
        if eef_pos is not None:
            target_pos = eef_pos + offset
            self.set_object_position(object_name, target_pos)
    
    def move_joints_sinusoidal(self, joint_indices, amplitudes, frequencies, step):
        """
        Move specified joints in a sinusoidal pattern.
        
        Args:
            joint_indices (list): Indices of joints to move (0-based index into robot joints)
            amplitudes (list): Amplitude of sinusoidal movement for each joint
            frequencies (list): Frequency of sinusoidal movement for each joint
            step (int): Current time step
        """
        if len(joint_indices) != len(amplitudes) or len(joint_indices) != len(frequencies):
            raise ValueError("joint_indices, amplitudes, and frequencies must have the same length")
        
        positions = self.get_joint_positions()
        
        for i, joint_idx in enumerate(joint_indices):
            if joint_idx < len(positions):
                positions[joint_idx] = amplitudes[i] * np.sin(step * frequencies[i])
        
        self.set_joint_positions(positions)
    
    def step_simulation(self):
        """
        Step the simulation forward with a zero action.
        """
        action = np.zeros(self.action_dim)
        self.env.step(action)
    
    def print_robot_info(self):
        """
        Print information about the robot joints and end-effector.
        """
        print("Robot joints:")
        for i, joint_idx in enumerate(self.robot_joint_indices):
            joint_name = self.sim.model.joint_id2name(joint_idx)
            print(f"Joint {i}: {joint_name}, qpos address: {self.robot_qpos_addr[i]}")
        
        print("\nGripper joints:")
        for i, joint_idx in enumerate(self.gripper_indices):
            joint_name = self.sim.model.joint_id2name(joint_idx)
            print(f"Gripper joint {i}: {joint_name}")
        
        print("\nEnd-effector position:", self.get_eef_position())
    
    def print_object_info(self):
        """
        Print information about the objects in the environment.
        """
        print("Objects:")
        for name, body_id in self.object_ids.items():
            pos = self.sim.data.body_xpos[body_id]
            print(f"{name} position: {pos}")
    
    def get_gripper_state(self):
        """
        Get the current state of the gripper.
        
        Returns:
            float: Current gripper width/opening
        """
        # For most robots in robosuite, the gripper state can be accessed through the robot's gripper property
        if hasattr(self.robot, 'gripper'):
            return self.robot.gripper.get_state()
        
        # Alternative method: directly access gripper joint positions if available
        if self.gripper_indices:
            qpos = self.sim.data.qpos.copy()
            gripper_pos = []
            for joint_idx in self.gripper_indices:
                addr = self.sim.model.jnt_qposadr[joint_idx]
                gripper_pos.append(qpos[addr])
            return np.array(gripper_pos)
        
        return None
    
    def set_gripper(self, value):
        """
        Set the gripper opening directly using an action.
        
        Args:
            value (float): Gripper control value. Negative to close, positive to open.
        """
        # Create an action with zeros for arm control and the specified value for gripper
        action = np.zeros(self.action_dim)
        
        # In robosuite, the last element of the action vector typically controls the gripper
        action[-1] = value
        
        # Apply the action
        self.env.step(action)
    
    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        """
        Compute inverse kinematics to find joint positions for a target end-effector pose.
        Uses a numerical optimization approach to find joint positions that achieve
        the desired end-effector position and orientation.
        
        Args:
            target_position (numpy.ndarray): Target 3D position for the end-effector
            target_orientation (numpy.ndarray, optional): Target orientation for the end-effector
            
        Returns:
            numpy.ndarray: Array of joint positions or None if failed
        """
        # Store initial state to restore if needed
        initial_qpos = self.sim.data.qpos.copy()
        initial_qvel = self.sim.data.qvel.copy()
        
        # Get current joint positions
        current_joint_positions = self.get_joint_positions()
        
        # Parameters for IK solver
        max_iterations = 100
        tolerance = 0.01
        step_size = 0.05
        damping = 0.1  # Damping factor for numerical stability
        
        # Initialize best solution tracking
        best_error = float('inf')
        best_joints = current_joint_positions.copy()
        
        # Iterate to find a solution
        for iteration in range(max_iterations):
            # Get current end-effector position
            current_eef_position = self.get_eef_position()
            
            # Calculate position error
            position_error = target_position - current_eef_position
            position_error_magnitude = np.linalg.norm(position_error)
            
            # Check if we're close enough
            if position_error_magnitude < tolerance:
                print(f"IK converged after {iteration} iterations")
                return current_joint_positions
            
            # Track best solution so far
            if position_error_magnitude < best_error:
                best_error = position_error_magnitude
                best_joints = current_joint_positions.copy()
            
            # Calculate Jacobian (approximation using finite differences)
            jacobian = np.zeros((3, len(current_joint_positions)))
            
            for j in range(len(current_joint_positions)):
                # Save current position
                original_pos = current_joint_positions[j]
                
                # Small perturbation
                delta = 0.001
                
                # Forward perturbation
                perturbed_positions = current_joint_positions.copy()
                perturbed_positions[j] = original_pos + delta
                self.set_joint_positions(perturbed_positions)
                forward_eef_position = self.get_eef_position()
                
                # Calculate column of Jacobian
                jacobian[:, j] = (forward_eef_position - current_eef_position) / delta
                
                # Restore original position
                current_joint_positions[j] = original_pos
                self.set_joint_positions(current_joint_positions)
            
            # Compute pseudo-inverse of Jacobian with damping for numerical stability
            # J† = J^T * (J * J^T + λI)^-1
            j_transpose = jacobian.T
            regularization = damping * np.eye(3)
            j_pseudo_inv = j_transpose.dot(
                np.linalg.inv(jacobian.dot(j_transpose) + regularization)
            )
            
            # Update joint positions using damped least squares method
            update = step_size * j_pseudo_inv.dot(position_error)
            current_joint_positions += update
            
            # Apply joint limits if needed
            # This is a simplified approach - implement proper joint limits if available
            joint_limits_lower = -3.14 * np.ones_like(current_joint_positions)
            joint_limits_upper = 3.14 * np.ones_like(current_joint_positions)
            current_joint_positions = np.clip(
                current_joint_positions, 
                joint_limits_lower, 
                joint_limits_upper
            )
            
            # Set the new joint positions
            self.set_joint_positions(current_joint_positions)
        
        print(f"IK did not fully converge after {max_iterations} iterations")
        print(f"Best achieved error: {best_error}")
        
        # Restore original state if error is too large
        if best_error > 0.1:  # 10cm error is too much
            self.sim.data.qpos[:] = initial_qpos
            self.sim.data.qvel[:] = initial_qvel
            self.sim.forward()
            print("Restoring original state due to large IK error")
            return None
        
        # Set the best solution found
        self.set_joint_positions(best_joints)
        return best_joints
    
    def move_to_position(self, target_position):
        """
        Move the robot's end-effector to a target position.
        Uses direct position control through the action space.
        
        Args:
            target_position (numpy.ndarray): Target 3D position for the end-effector
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Get current end-effector position
        current_eef_position = self.get_eef_position()
        
        # Calculate position error
        position_error = target_position - current_eef_position
        
        # Create an action that moves towards the target
        action = np.zeros(self.action_dim)
        
        # Scale the position error to create an appropriate action
        # The first 3 elements typically control xyz position
        position_gain = 2.0  # Lower gain for smoother movement
        action[:3] = position_gain * position_error
        
        # Apply the action
        self.env.step(action)
        
        # Return success based on how close we got to the target
        new_eef_position = self.get_eef_position()
        final_error = np.linalg.norm(target_position - new_eef_position)
        
        # Print info less frequently to avoid console spam
        if np.random.random() < 0.05:  # Only print ~5% of the time
            print(f"Moving to target position: {target_position}")
            print(f"Current position: {new_eef_position}")
            print(f"Position error: {final_error}")
        
        return final_error < 0.1  # Consider it successful if within 10cm
    
    def set_orientation(self, target_orientation):
        """
        Set the robot's end-effector to a target orientation.
        Uses direct orientation control through the action space.
        
        Args:
            target_orientation (numpy.ndarray): Target orientation as quaternion [w, x, y, z]
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Get current end-effector orientation
        current_eef_orientation = self.get_eef_orientation()
        
        if current_eef_orientation is None:
            print("Warning: Could not get current end-effector orientation")
            return False
        
        # Create an action that rotates towards the target
        action = np.zeros(self.action_dim)
        
        # Normalize quaternions
        current_q = current_eef_orientation / np.linalg.norm(current_eef_orientation)
        target_q = target_orientation / np.linalg.norm(target_orientation)
        
        # Calculate dot product (cosine of half the rotation angle)
        dot_product = np.abs(np.sum(current_q * target_q))
        
        # If quaternions are close to opposite, flip one
        if dot_product < 0:
            target_q = -target_q
            dot_product = -dot_product
        
        # Calculate rotation error
        # The smaller the dot product, the larger the rotation needed
        rotation_error = 1.0 - dot_product
        
        # Apply rotation control (simplified)
        # In robosuite, elements 3-6 typically control rotation
        if rotation_error > 0.01:  # Only apply if there's significant error
            rotation_gain = 1.0
            # This is a simplified approach - in a real system, you'd calculate
            # the exact rotation axis and angle
            if len(action) > 6:  # Make sure action vector has rotation components
                action[3:6] = rotation_gain * rotation_error * (target_q[1:4] - current_q[1:4])
        
        # Apply the action
        self.env.step(action)
        
        # Return success based on how close we got to the target
        new_eef_orientation = self.get_eef_orientation()
        
        if new_eef_orientation is None:
            return False
        
        # Normalize for comparison
        new_q = new_eef_orientation / np.linalg.norm(new_eef_orientation)
        
        # Calculate final error
        final_dot_product = np.abs(np.sum(new_q * target_q))
        final_error = 1.0 - final_dot_product
        
        # Print info less frequently to avoid console spam
        if np.random.random() < 0.05:  # Only print ~5% of the time
            print(f"Setting orientation to: {target_orientation}")
            print(f"Current orientation: {new_eef_orientation}")
            print(f"Orientation error: {final_error}")
        
        return final_error < 0.1  # Consider it successful if within threshold
    
    def move_to_pose(self, target_position, target_orientation=None):
        """
        Move the robot's end-effector to a target pose.
        This is a convenience method that combines position and orientation control.
        
        Args:
            target_position (numpy.ndarray): Target 3D position for the end-effector
            target_orientation (numpy.ndarray, optional): Target orientation for the end-effector
        
        Returns:
            bool: True if successful, False otherwise
        """
        # First move to the target position
        position_success = self.move_to_position(target_position)
        
        # Then set the orientation if provided
        if target_orientation is not None:
            orientation_success = self.set_orientation(target_orientation)
            return position_success and orientation_success
        
        return position_success
