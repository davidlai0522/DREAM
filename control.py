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
        ignore_done=True,
    )
    
    # Reset the environment
    env.reset()
    
    # Create the robot controller
    controller = RobotController(env)
    
    # Print robot and object information
    controller.print_robot_info()
    controller.print_object_info()
    
    # Render loop
    done = False
    step = 0
    
    while step < 200:
        target_positions = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.8, 0.0]) # [j1, j2, j3, j4, j5, j6, j7]
        controller.set_joint_positions(target_positions)
        controller.attach_object_to_eef("RoundNut")
        print(f"Step {step}: Set joints to target positions")
        # Step the simulation
        controller.step_simulation()
        env.render()
        step += 1
        time.sleep(0.2)
        
    step = 0
    while step < 5000 and not done:
        cycle = step % 1000
        
        if cycle < 200:  # First position
            target_position = np.array([0.25, 0.0, 1.3])
            controller.move_to_position(target_position)
            if cycle == 0:
                print(f"Step {step}: Moving to position 1 (x=0.25, y=0.0, z=1.3)")
                
        elif cycle < 400:  # Second position
            target_position = np.array([0.25, 0.3, 1.4])
            controller.move_to_position(target_position)
            if cycle == 200:
                print(f"Step {step}: Moving to position 2 (x=0.25, y=0.3, z=1.4)")
                
        elif cycle < 600:  # Third position
            target_position = np.array([0.3, -0.3, 1.4])
            controller.move_to_position(target_position)
            if cycle == 400:
                print(f"Step {step}: Moving to position 3 (x=0.3, y=-0.3, z=1.4)")
                
        elif cycle < 700:  # Use inverse kinematics to move to a position
            if cycle == 600 or cycle % 50 == 0:
                target_position = np.array([0.4, 0.2, 1.5])
                # Compute joint positions using IK
                joint_positions = controller.compute_inverse_kinematics(target_position)
                if joint_positions is not None:
                    controller.set_joint_positions(joint_positions)
                    print(f"Step {step}: Using IK to move to position (x=0.4, y=0.2, z=1.5)")
                else:
                    print(f"Step {step}: IK failed, using direct position control instead")
                    controller.move_to_position(target_position)
                
        elif cycle < 800:  # Change orientation 2
            target_orientation = np.array([0.7071, 0.7071, 0, 0])  # 90 deg around x
            controller.set_orientation(target_orientation)
            if cycle == 700:
                print(f"Step {step}: Setting orientation 2 (90° around x-axis)")
                
        elif cycle < 900:  # Change orientation 3
            target_orientation = np.array([0.7071, 0, 0.7071, 0])  # 90 deg around y
            controller.set_orientation(target_orientation)
            if cycle == 800:
                print(f"Step {step}: Setting orientation 3 (90° around y-axis)")
                
        else:  # Combined position and orientation
            target_position = np.array([0.15, 0.0, 1.5])
            target_orientation = np.array([0.7071, 0, 0, 0.7071])  # 90 deg around z
            controller.move_to_pose(target_position, target_orientation)
            if cycle == 900:
                print(f"Step {step}: Moving to position 4 with orientation 4 (90° around z-axis)")
        
        # Toggle gripper every 100 steps
        if step % 100 < 50:
            controller.set_gripper(-0.1)  # Open gripper
        else:
            controller.set_gripper(0.1)  # Close gripper
        
        # Step the simulation
        controller.step_simulation()
        env.render()
        
        time.sleep(0.02)  # Add small delay to make rendering visible
        step += 1
    
    env.close()
