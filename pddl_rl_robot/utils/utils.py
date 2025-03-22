import numpy as np

def quat2euler(quat: np.ndarray, use_degree: bool = False) -> np.ndarray:
    euler = np.array([
        round(np.arctan2(2*(quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2)), 2),
        round(np.arcsin(2*(quat[0]*quat[2] - quat[3]*quat[1])), 2),
        round(np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]), 1 - 2*(quat[2]**2 + quat[3]**2)), 2)
    ])
    if use_degree:
        euler = np.degrees(euler)
    return euler