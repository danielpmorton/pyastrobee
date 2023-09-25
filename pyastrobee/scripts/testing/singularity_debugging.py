"""WIP: Script to help debug a rotation singularity within pybullet

Right not this script will just reproduce the problem

Run this with the "OK" poses to see what the rotation and control inputs SHOULD look like
"""

import numpy as np
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.config.astrobee_motion import MAX_FORCE_MAGNITUDE, MAX_TORQUE_MAGNITUDE
from pyastrobee.trajectories.quaternion_interpolation import (
    quaternion_interpolation_with_bcs,
)
from pyastrobee.utils.quaternions import quats_to_angular_velocities
from pyastrobee.trajectories.trajectory import Trajectory

# Note for reference: these work great (These are just two random quaternions that I liked)
# fmt: off
# start_pose = [0, 0, 0, 0.47092437141245563, 0.8134303596981151, 0.0001291583857834318, 0.3414107052353368]
# end_pose = [0, 0, 0, 0.34196961983991075, 0.21516679142364756, 0.4340223300167253, 0.8052233528790954]
# fmt: on

# This is problematic
start_pose = [0, 0, 0, 0, 0, 1, 0]
end_pose = [0, 0, 0, 0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2]

client = initialize_pybullet()
dt = client.getPhysicsEngineParameters()["fixedTimeStep"]

robot = Astrobee(start_pose)
robot.store_arm(force=True)
controller = ForceTorqueController(
    robot.id,
    robot.mass,
    robot.inertia,
    20,
    5,
    1,
    0.1,
    dt,
    max_force=MAX_FORCE_MAGNITUDE * 10,  # TODO REMOVE
    max_torque=MAX_TORQUE_MAGNITUDE * 10,
)
tf = 10
n_timesteps = round(tf / dt)
quats = quaternion_interpolation_with_bcs(
    start_pose[3:],
    end_pose[3:],
    np.zeros(3),
    np.zeros(3),
    np.zeros(3),
    np.zeros(3),
    tf,
    n_timesteps,
)
omega = quats_to_angular_velocities(quats, dt)
alpha = np.gradient(omega, dt, axis=0)
traj = Trajectory(
    start_pose[:3] * np.ones((n_timesteps, 1)),
    quats,
    np.zeros((n_timesteps, 3)),
    omega,
    np.zeros((n_timesteps, 3)),
    alpha,
    np.arange(n_timesteps) * dt,
)

traj.plot()
controller.follow_traj(traj, False)
controller.traj_log.plot()
controller.control_log.plot()
