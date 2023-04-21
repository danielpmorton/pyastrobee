"""Simple script to give an example of creating and visualziing trajectories in Pybullet"""

import time

import numpy as np
import pybullet

from pyastrobee.control.trajectory import visualize_traj
from pyastrobee.control.planner import interpolation_pose_traj, point_and_move_pose_traj
from pyastrobee.control.polynomial_trajectories import polynomial_trajectory
from pyastrobee.utils.quaternion import random_quaternion

# Create start/end poses
np.random.seed(0)
start_pos = np.array([-3, 0, 1])
end_pos = np.array([3, 0, 1])
start_orn = random_quaternion()
end_orn = random_quaternion()
start_pose = np.concatenate((start_pos, start_orn))
end_pose = np.concatenate((end_pos, end_orn))

# Make trajectories with different methods, with position offset for easier visualization
dp = np.array([0, -2, 0, 0, 0, 0, 0])
traj_1 = interpolation_pose_traj(start_pose, end_pose, 30)
traj_2 = point_and_move_pose_traj(start_pose + dp, end_pose + dp, 0.2, 0.2)
traj_3 = polynomial_trajectory(start_pose - dp, end_pose - dp, 5, 0.1)

# Visualize the trajectories in Pybullet
pybullet.connect(pybullet.GUI)
ids_1 = visualize_traj(traj_1)
pybullet.addUserDebugText("Pose interpolation", start_pos + [0, 0, 1])
ids_2 = visualize_traj(traj_2)
pybullet.addUserDebugText("Point-and-move", start_pos + [0, -2, 1])
ids_3 = visualize_traj(traj_3)
pybullet.addUserDebugText("3rd-order polynomial", start_pos + [0, 2, 1])


# Leave the sim running
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 120)
