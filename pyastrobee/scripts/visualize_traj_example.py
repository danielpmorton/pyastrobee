"""Simple script to give an example of visualizing a trajectory in Pybullet"""

import time

import numpy as np
import pybullet

from pyastrobee.utils.rotations import rmat_to_quat, Rz
from pyastrobee.vision.debug_visualizer import visualize_traj
from pyastrobee.utils.rotations import quaternion_slerp

# Make an arbitrary trajectory just as an example:

# Spherical linear interpolation for orientation component
q1 = np.array([0, 0, 0, 1])
q2 = rmat_to_quat(Rz(np.pi / 2))
n_steps = 50
t = np.linspace(0, 1, n_steps)
xyzw_quats = quaternion_slerp(q1, q2, t)

# Linear interpolation for position component
p1 = np.array([0, 0, 0])
p2 = np.array([10, 0, 0])
positions = p1 + p2 * t.reshape(-1, 1)

# Form the (n, 7) position + quaternion trajectory
traj = np.hstack((positions, xyzw_quats))

# Visualize the trajectory in Pybullet
pybullet.connect(pybullet.GUI)
ids = visualize_traj(traj)

# Leave the sim running
while True:
    pybullet.stepSimulation()
    time.sleep(1 / 120)
