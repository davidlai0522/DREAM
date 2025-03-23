import numpy as np
from pddl_rl_robot.deterministic.deterministic_base_class import DeterministicBaseClass


class LiftAction(DeterministicBaseClass):
    """
    Deterministic action to lift an object after it has been grasped.
    
    This class implements a simple policy that moves the robot arm upward
    while keeping the gripper closed to lift a grasped object.
    """
    
    def __init__(self, env, gripper_close_value=1.0, lift_height=0.2):
        """
        Initialize the lift action.
        
        Args:
            env: The environment the policy will interact with
            gripper_close_value: Value to keep the gripper closed (-1.0 typically closes the gripper)
            lift_height: Height to lift the object (in meters)
        """
        super().__init__(env)
        self.gripper_close_value = gripper_close_value
        self.lift_height = lift_height
        self.lift_steps = 0
        self.max_lift_steps = 50  # Number of steps to complete the lift action
        
        # For Panda robot, the joint that controls vertical movement is typically joint 1 (shoulder)
        # and joint 3 (elbow). We'll focus on these joints for lifting.
        self.shoulder_joint_idx = 1  # Typically controls up/down movement
        self.elbow_joint_idx = 3     # Also affects height
        self.wrist_joint_idx = 5     # Sixth joint (wrist) that we want to keep stable
        
        # Additional joints that affect orientation
        self.orientation_joints = [4, 5, 6]  # Typically joints 5, 6, 7 (0-indexed as 4, 5, 6) control orientation
        
        # Initialize variables for inverse kinematics solver
        self.ik_solver_initialized = False
        self.target_position = None
        self.start_position = None
        self.interpolation_fraction = 0.0
        self.interpolation_step = 0.2  # Controls the speed of lifting
        
        # Get the action space dimension (excluding gripper)
        self.action_dim = self.env.action_space.shape[0] - 1
        print(f"Action space dimension (excluding gripper): {self.action_dim}")

        # Skip IK solver for now - use direct joint control
        self.use_ik_solver = False
        
        # Find the correct grip site name from the available options
        self.grip_site_name = self._find_grip_site_name()
        if self.grip_site_name:
            print(f"Using grip site: {self.grip_site_name}")
        else:
            print("WARNING: Could not find a suitable grip site, falling back to joint-based control")
            self.use_ik_solver = False
            
        # Store initial joint positions to maintain orientation
        self._initial_joint_positions = None

    def _find_grip_site_name(self):
        """Find an appropriate grip site name from available sites in the model."""
        try:
            # List of potential grip site names, in order of preference
            potential_names = [
                'gripper0_right_grip_site',  # Based on error message
                'robot0_grip_site',          # Original attempt
                'grip_site',                 # Common alternative
                'end_effector',              # Another common name
                'ee_site'                    # Another common name
            ]
            
            # Try to find one that exists in the model
            for name in potential_names:
                try:
                    self.env.sim.model.site_name2id(name)
                    return name
                except:
                    continue
                    
            # If none found, print available sites for debugging
            sites = []
            for i in range(self.env.sim.model.nsite):
                site_name = self.env.sim.model.site_id2name(i)
                sites.append(site_name)
                
            print(f"Available sites: {sites}")
            return None
            
        except Exception as e:
            print(f"Error finding grip site: {e}")
            return None

    # Get the initial end effector position when the action starts
    def _initialize_ik_solver(self):
        """Initialize the inverse kinematics solver with current robot position."""
        try:
            if not self.grip_site_name:
                print("No grip site found. Cannot initialize IK solver.")
                self.use_ik_solver = False
                return False
                
            # Get current end effector position
            ee_site_id = self.env.sim.model.site_name2id(self.grip_site_name)
            self.start_position = self.env.sim.data.site_xpos[ee_site_id].copy()
            
            # Set target position higher than the start position
            self.target_position = self.start_position.copy()
            self.target_position[2] += self.lift_height  # Increase Z coordinate
            
            self.interpolation_fraction = 0.0
            self.ik_solver_initialized = True
            print(f"IK solver initialized. Start position: {self.start_position}, Target position: {self.target_position}")
            
            # Store initial joint positions for orientation maintenance
            try:
                self._initial_joint_positions = self.env.sim.data.qpos.copy()
                print(f"Initial joint positions stored for orientation maintenance")
            except Exception as e:
                print(f"Error storing initial joint positions: {e}")
            
            # Set to use joint control by default since IK is causing issues
            self.use_ik_solver = False
            return True
        except Exception as e:
            print(f"Error initializing IK solver: {e}")
            # Fall back to joint-based approach
            self.ik_solver_initialized = False
            self.use_ik_solver = False
            return False
    
    def _fallback_joint_control(self):
        """
        Fallback method using joint-based control for lifting while preserving orientation.
        """
        # Create a simple joint movement vector for the action space
        dq = np.zeros(self.action_dim)
        
        # Calculate progress factor (0.0 to 1.0)
        progress_factor = min(self.lift_steps / 50.0, 1.0)
        
        # Simple hardcoded joint movement that typically causes lifting
        if self.shoulder_joint_idx < self.action_dim:
            # Shoulder joint - positive moves arm upward
            dq[self.shoulder_joint_idx] = 0.05 * progress_factor
        
        if self.elbow_joint_idx < self.action_dim:
            # Elbow joint - negative extends arm upward
            dq[self.elbow_joint_idx] = -0.03 * progress_factor
        
        # Explicitly ensure all orientation joints don't change
        for joint_idx in self.orientation_joints:
            if joint_idx < self.action_dim:
                dq[joint_idx] = 0.0  # Explicitly set to zero to maintain current angle
                
        # If we detect orientation drift, try to correct it
        if self._initial_joint_positions is not None and self.lift_steps > 5:
            try:
                current_joints = self.env.sim.data.qpos.copy()
                # Check orientation joints for drift
                for joint_idx in self.orientation_joints:
                    if joint_idx < len(current_joints) and joint_idx < len(self._initial_joint_positions):
                        # Calculate the drift from initial position
                        drift = current_joints[joint_idx] - self._initial_joint_positions[joint_idx]
                        # If drift is significant, apply a small correction (with proper array index adjustment)
                        if abs(drift) > 0.05 and joint_idx < self.action_dim:
                            dq[joint_idx] = -0.1 * drift  # Apply a correction proportional to the drift
                            # print(f"Correcting orientation drift in joint {joint_idx}: {drift}")
            except Exception as e:
                print(f"Error during orientation correction: {e}")
            
        return dq
    
    def _get_action(self, observation):
        """
        Generate an action that moves the robot arm upward while keeping the gripper closed.
            
        Returns:
            action: An action vector with joint movements for lifting and closed gripper
        """
        # Initialize a zero action vector
        action = np.zeros(self.env.action_space.shape)
        
        # Keep the gripper closed
        action[-1] = self.gripper_close_value
        
        # Update lift progress
        self.lift_steps += 1
        
        # Initialize the IK solver if it's not already initialized
        if not self.ik_solver_initialized:
            self._initialize_ik_solver()
        
        # Use joint control approach (since IK is causing issues)
        action[:-1] = self._fallback_joint_control()
        
        # Print debug information
        if self.lift_steps % 20 == 0:
            try:
                if self.grip_site_name:
                    ee_site_id = self.env.sim.model.site_name2id(self.grip_site_name)
                    current_pos = self.env.sim.data.site_xpos[ee_site_id]
                    print(f"Lift step {self.lift_steps}, EE position: {current_pos}")
                    
                    # Add orientation debugging if possible
                    try:
                        # Extract current orientation from simulation if available
                        site_mat = self.env.sim.data.site_xmat[ee_site_id].reshape(3, 3)
                        # Convert to a more readable format (e.g., Z-axis direction)
                        z_axis = site_mat[:, 2]
                        # print(f"Current EE orientation (Z-axis): {z_axis}")
                    except Exception as e:
                        pass  # Silently ignore orientation debugging errors
                else:
                    pass
                    # print(f"Lift step {self.lift_steps}, using joint control")
            except Exception as e:
                # print(f"Lift step {self.lift_steps}, error getting position: {e}")
                pass
        # Final check to ensure the gripper remains closed
        action[-1] = self.gripper_close_value
        
        # Store the current action for potential future use
        self._previous_action = action.copy()
        
        return action