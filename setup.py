from setuptools import setup, find_packages

setup(
    name="pddl_rl_robot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gymnasium",
        "torch",
        "pddlpy",
        "mujoco",
        "stable-baselines3",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for PDDL planning with RL-based robot execution",
    keywords="pddl, reinforcement learning, robotics",
    python_requires=">=3.8",
)
