"""Script to give an example of trajectory tracking using a unit-mass cube

This can also be used as a sandbox for experimenting with the PID tuning
"""

import pybullet
import numpy as np

from pyastrobee.control.force_controller_new import ForcePIDController
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.control.polynomial_trajectories import polynomial_trajectory
from pyastrobee.control.trajectory import (
    visualize_traj,
    compare_trajs,
    TrajectoryLogger,
)


def box_inertia(m, l, w, h):
    return (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])


tracker = TrajectoryLogger()
pybullet.connect(pybullet.GUI)
np.random.seed(0)
pose_1 = [0, 0, 0, 0, 0, 0, 1]
pose_2 = [1, 2, 3, *random_quaternion()]
mass = 10
sidelengths = [0.25, 0.25, 0.25]
box = create_box(pose_1[:3], pose_1[3:], mass, sidelengths, True)
max_time = 10
dt = 0.01
traj = polynomial_trajectory(pose_1, pose_2, max_time, dt)
visualize_traj(traj, 20)
# mass * dt seems to give a general trend of how the required gains change depending on mass/time
# However, it seems like this shouldn't depend on dt? Perhaps it's an artifact of doing discrete simulation steps
kp = 1000 * mass * dt
kv = 100 * mass * dt
kq = 10 * mass * dt
kw = 1 * mass * dt
base_idx = -1  # Base link index of the robot
inertia = box_inertia(mass, *sidelengths)
controller = ForcePIDController(box, mass, inertia, kp, kv, kq, kw, dt)
controller.follow_traj(traj)
pybullet.disconnect()
compare_trajs(traj, controller.traj_log)
controller.control_log.plot()
