"""

WORK IN PROGRESS

References:
nngusto: https://stanfordasl.github.io/wp-content/papercite-data/pdf/Banerjee.Lew.Bonalli.ea.AeroConf20.pdf
aoude: http://dspace.mit.edu/bitstream/handle/1721.1/42050/230816006-MIT.pdf?sequence=2&isAllowed=y
ahrs: https://ahrs.readthedocs.io/en/latest/filters/angular.html

aoude has an alternate way of representing the quaternion derivative (he forms the matrix based on the quaternion
rather than the angular velocity) but it gives the same result. See page 59, eqns 2.47/48
"""
import numpy as np
from pyastrobee.utils.quaternion import random_quaternion
from pyastrobee.utils.math_utils import skew


# # See nngusto
# def S(w):
#     return skew(w)


# See nngusto
# Modified to suit XYZW quats rather than WXYZ
# nasa/astrobee seems to have this as well (ctl.cc) but they did not seem to modify to suit XYZW??
def Omega(w):
    w = np.asarray(w)
    wx, wy, wz = w
    return np.array(
        [[wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0], [0, -wx, -wy, -wz]]
    )


# def O_new(w):
#     w = np.ravel(w).reshape(-1, 1)
#     return np.block([[w, skew(w)], [0, -w.T]])


def quaternion_derivative(q, w):
    assert len(w) == 3
    assert len(q) == 4
    return (1 / 2) * Omega(w) @ q


def quaternion_delta(q, w, dt):
    return dt * quaternion_derivative(q, w)


def quaternion_integration(q, w, dt):  # is this needed?
    # return (np.eye(4) + (1 / 2) * Omega(w) * dt) @ q
    return q + dt * quaternion_derivative(q, w)


# This doesn't seem to actually normalize the output in a closed-form manner.... hmmm probably not useful
def normalized_quaternion_integration(q, w, dt):
    w_norm = np.linalg.norm(w)
    return (
        np.cos(w_norm * dt / 2) * np.eye(4)
        + (1 / w_norm) * np.sin(w_norm * dt / 2) * Omega(w)
    ) @ q


# from nngusto
def angular_accel(torque, ang_vel, inertia):
    return np.linalg.inv(inertia) @ (torque - skew(ang_vel) @ inertia @ ang_vel)


def ang_vel_integration(torque, ang_vel, inertia, dt):
    return ang_vel + dt * angular_accel(torque, ang_vel, inertia)


# Aoude page 59, 2.47/48
# def Z(q):
#     q = np.asarray(q)
#     qx, qy, qz, qw = q
#     return np.array([[qw, -qz, qy], [qz, qw, -qx], [-qy, qx, qw], [-qx, -qy, -qz]])
# def quaternion_derivative(q, w):
#     # Aoude method
#     return (1 / 2) * Z(w) @ w

q = random_quaternion()
w = np.random.rand(3) * 10
dt = 0.1
q_new_1 = quaternion_integration(q, w, dt)
q_new_2 = normalized_quaternion_integration(q, w, dt)
print(q_new_1)
print(q_new_2)
print(np.linalg.norm(q_new_1))
print(np.linalg.norm(q_new_2))
print(q_new_1 / np.linalg.norm(q_new_1))
print(q_new_2 / np.linalg.norm(q_new_2))
