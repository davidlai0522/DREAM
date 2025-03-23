import time
from pddl_rl_robot.simulation.two_peg_one_disk_env import TwoPegOneRoundNut
import numpy as np
from robot_controller import RobotController

if __name__ == "__main__":
    # Create environment instance
    env = TwoPegOneRoundNut(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )
    
    # Reset the environment
    env.reset()
    
    # Create the robot controller
    controller = RobotController(env)
    
    # Print robot and object information
    controller.print_robot_info()
    controller.print_object_info()
    
    # # Set robot to a specific joint configuration
    # initial_joint_positions = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8])
    # controller.set_joint_positions(initial_joint_positions)
    # time.sleep(1)  # Give time to see the movement
    
    
    # Render loop
    done = False
    step = 0
    
    while step < 1000 and not done:
        # if step < 100:
        #     # Example 1: Move joints with sinusoidal pattern
        #     controller.move_joints_sinusoidal(
        #         joint_indices=[0, 1, 3],  # First, second, and fourth joints
        #         amplitudes=[0.5, 0.3, 0.4],  # Amplitudes for each joint
        #         frequencies=[0.05, 0.03, 0.04],  # Frequencies for each joint
        #         step=step
        #     )
        # else:
        # Example 2: set all joints to specific positions
        target_positions = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.8, 0.0])
        controller.set_joint_positions(target_positions)
        controller.attach_object_to_eef("RoundNut")
        print(f"Step {step}: Set joints to target positions")
        
        # # Example 3: Every 300 steps, attach the RoundNut to the end-effector
        # if step % 300 == 0:
        #     controller.attach_object_to_eef("RoundNut")
        #     print(f"Step {step}: Attached RoundNut to end-effector")
        
        # # Every 50 steps, print the positions
        # if step % 50 == 0:
        #     ee_pos = controller.get_eef_position()
        #     nut_pos = controller.get_object_position("RoundNut")
        #     print(f"Step {step}: End-effector position: {ee_pos}")
        #     print(f"RoundNut position: {nut_pos}")
        
        # Step the simulation
        controller.step_simulation()
        env.render()
        
        time.sleep(0.2)  # Add small delay to make rendering visible
        step += 1
    
    env.close()
