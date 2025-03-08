import time
from pddl_rl_robot.simulation.simulator import RobotSimulator

sim = RobotSimulator(
    robot_type="Panda",
    env_name="two_peg_one_disk",
)

obs, _ = sim.reset()

print(f"The action space of Panda Robot is: {sim.gym_env.action_space}")
print(f"The observation space of Panda Robot is: {sim.gym_env.observation_space}")


for i in range(1000):
    action = sim.get_random_action()
    print(f"Step {i}, Action: {action}")
    try:
        # Handle both 4-value and 5-value return formats
        result = sim.step(action)
        if len(result) == 5:  # New format (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old format (obs, reward, done, info)
            obs, reward, done, info = result
            
        sim.render()

        if done:
            print(f"Episode done at step {i}. Resetting environment...")
            
    except KeyboardInterrupt:
        print("Interrupted by user.")
        break

for i in range(100):
    action = sim.get_no_action()
    print(f"Step {i}, Action: {action}")
    try:
        # Handle both 4-value and 5-value return formats
        result = sim.step(action)
        if len(result) == 5:  # New format (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old format (obs, reward, done, info)
            obs, reward, done, info = result
            
        sim.render()

        if done:
            print(f"Episode done at step {i}. Resetting environment...")
            
    except KeyboardInterrupt:
        print("Interrupted by user.")
        break

time.sleep(10)
sim.close()
exit()