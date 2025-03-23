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
        
        print("\nEnd-effector position:", self.get_eef_position())
    
    def print_object_info(self):
        """
        Print information about the objects in the environment.
        """
        print("Objects:")
        for name, body_id in self.object_ids.items():
            pos = self.sim.data.body_xpos[body_id]
            print(f"{name} position: {pos}")
            
            
    def inverse_kinematics(self, target_position, target_orientation=None):
        """
        Compute inverse kinematics to reach a target position and orientation.
        
        Args:
            target_position (numpy.ndarray): 3D target position
            target_orientation (numpy.ndarray): 4D target orientation (quaternion)
            
        Returns:
            numpy.ndarray: Array of joint positions
        """
        return self.robot.inverse_kinematics(target_position, target_orientation)
