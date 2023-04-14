"""Simple control examples using PID/LQR/..."""
import time
import pybullet
import numpy as np
from pyastrobee.control.pid import PID
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.quaternion import random_quaternion
from pyastrobee.utils.rotations import quat_to_fixed_xyz


def get_state(id):
    # Assume our state vector is [x, y, z, roll, pitch, yaw, vx, vy, vz, v_roll, v_pitch, v_yaw]
    pos, quat = pybullet.getBasePositionAndOrientation(id)
    # Use fixed XYZ convention for the orientation
    # (This aligns with how pybullet defines the angular velocity vector)
    orn = quat_to_fixed_xyz(quat)
    vel, ang_vel = pybullet.getBaseVelocity(id)
    return np.array([*pos, *orn, *vel, *ang_vel])


def get_forces_and_torques(pid_command):
    # Assume we still have our size-12 state vector
    pass


# Start out with a super simple "robot"
# Just a cube for simple dynamics

pybullet.connect(pybullet.GUI)
pos = [0, 0, 1]
orn = random_quaternion()
mass = 1
sidelengths = [1, 1, 1]
use_collision = True
rgba = [1, 1, 1, 1]
cube_id = create_box(pos, orn, mass, sidelengths, use_collision, rgba)

# p_gains = np.array()
# i_gains = np.array()
# d_gains = np.array()
# i_min = None
# i_max = None
# controller = PID(p_gains, i_gains, d_gains, i_min, i_max)


# pybullet.calculateMassMatrix

dt = 1 / 120

while True:
    pybullet.stepSimulation()
    # state = get_state(cube_id)
    # des_state = trajectory[some_index]
    # error = des_state - state
    # command = controller.update(error, dt)

    time.sleep(dt)
