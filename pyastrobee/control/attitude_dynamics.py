"""
WORK IN PROGRESS

References:
nngusto: https://stanfordasl.github.io/wp-content/papercite-data/pdf/Banerjee.Lew.Bonalli.ea.AeroConf20.pdf
aoude: http://dspace.mit.edu/bitstream/handle/1721.1/42050/230816006-MIT.pdf?sequence=2&isAllowed=y
ahrs: https://ahrs.readthedocs.io/en/latest/filters/angular.html
Shuster: http://malcolmdshuster.com/Pub_1993h_J_Repsurv_scan.pdf

aoude has an alternate way of representing the quaternion derivative (he forms the matrix based on the quaternion
rather than the angular velocity) but it gives the same result. See page 59, eqns 2.47/48
"""
import numpy as np
import numpy.typing as npt
from pyastrobee.utils.quaternion import random_quaternion
from pyastrobee.utils.math_utils import skew
from pyastrobee.utils.transformations import transform_point, invert_transform_mat


def angular_accel(
    torque: npt.ArrayLike, ang_vel: npt.ArrayLike, inertia: np.ndarray
) -> np.ndarray:
    """Angular acceleration of a body when a torque is applied, in body frame

    Based on Aoude, 2005, eqn. 2.7 on pg. 34

    Args:
        torque (npt.ArrayLike): Applied torque on the body (body frame), shape (3,)
        ang_vel (npt.ArrayLike): Current angular velocity of the body (body frame), shape (3,)
        inertia (np.ndarray): Inertia matrix for the body, shape (3,3)

    Returns:
        np.ndarray: Angular acceleration, shape (3,)
    """
    return np.linalg.inv(inertia) @ (-1 * np.cross(ang_vel, inertia @ ang_vel) + torque)


# TODO
# Decide how to best handle the inputs and their frames
# Example:
# world_frame_ang_acc = transform_point(T_R2W, body_frame_ang_acc)
# T_R2W = invert_transform_mat(T_R2W)
# body_frame_ang_vel = transform_point(T_W2R, world_frame)
def world_frame_angular_accel(torque, ang_vel, inertia, T_R2W):
    pass


# Decide on frame definitions
def angular_velocity_integration(torque, ang_vel, inertia, dt):
    return ang_vel + dt * angular_accel(torque, ang_vel, inertia)
