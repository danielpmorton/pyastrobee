"""More rotation interpolation methods that naturally account for boundary conditions and guarantee quaternion 
normalization, unlike the quaternion polynomial method

These "spherical bezier curves" are relatively computationally intensive to solve, because it requires a lot of calls 
to SLERP for each point we're interpolating (due to the recursive nature of De Casteljau's algorithm)

In most cases it will be better to use fast approximate methods to rotation planning, but this can serve as a good
reference curve

TODO
- This file needs a lot of cleanup if we are in fact using some of these methods
    - Move quaternion math operations to the main quaternions file?
- Get nonzero angular acceleration vector BCs implemented (currently can only handle zero magnitude in accel)
- Check out the SQUAD algorithm. This seems very similar to a cubic decasteljau spherical bezier curve but with a
  different parameterization of the "t" interpolation value

Reference:
https://www.cs.cmu.edu/~kiranb/animation/p245-shoemake.pdf
https://web.archive.org/web/20120915153625/http://courses.cms.caltech.edu/cs171/quatut.pdf
https://splines.readthedocs.io/en/latest/rotation/index.html
"""

import pybullet
import numpy as np

from pyastrobee.utils.quaternions import (
    quaternion_slerp,
    random_quaternion,
    quats_to_angular_velocities,
    conjugate,
    exponential_map,
    log_map,
    pure,
    multiply,
)
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.debug_visualizer import animate_rotation

# Interpolation methods


def cubic_decasteljau_slerp(q0, q1, q2, q3, t):
    if not np.isscalar(t):
        # If t is a list, return a list of unit quaternions
        return [cubic_decasteljau_slerp(q0, q1, q2, q3, t_i) for t_i in t]
    slerp_01 = quaternion_slerp(q0, q1, t)
    slerp_12 = quaternion_slerp(q1, q2, t)
    slerp_23 = quaternion_slerp(q2, q3, t)
    return quaternion_slerp(
        quaternion_slerp(slerp_01, slerp_12, t),
        quaternion_slerp(slerp_12, slerp_23, t),
        t,
    )


def quintic_decasteljau_slerp(q0, q1, q2, q3, q4, q5, t):
    if not np.isscalar(t):
        # If t is a list, return a list of unit quaternions
        return [quintic_decasteljau_slerp(q0, q1, q2, q3, q4, q5, t_i) for t_i in t]
    slerp_01 = quaternion_slerp(q0, q1, t)
    slerp_12 = quaternion_slerp(q1, q2, t)
    slerp_23 = quaternion_slerp(q2, q3, t)
    slerp_34 = quaternion_slerp(q3, q4, t)
    slerp_45 = quaternion_slerp(q4, q5, t)

    slerp_012 = quaternion_slerp(slerp_01, slerp_12, t)
    slerp_123 = quaternion_slerp(slerp_12, slerp_23, t)
    slerp_234 = quaternion_slerp(slerp_23, slerp_34, t)
    slerp_345 = quaternion_slerp(slerp_34, slerp_45, t)

    slerp_0123 = quaternion_slerp(slerp_012, slerp_123, t)
    slerp_1234 = quaternion_slerp(slerp_123, slerp_234, t)
    slerp_2345 = quaternion_slerp(slerp_234, slerp_345, t)

    slerp_01234 = quaternion_slerp(slerp_0123, slerp_1234, t)
    slerp_12345 = quaternion_slerp(slerp_1234, slerp_2345, t)

    return quaternion_slerp(slerp_01234, slerp_12345, t)


# This parameterization of t is the same as in the Shoemake paper but it doesn't seem to make any sense,
# because it's a quadratic that doesn't actually reach 1, so we don't get to the final interpolated quaternion?
# Though, I might be doing this wrong
def squad(q0, q1, q2, q3, t):
    return cubic_decasteljau_slerp(q0, q1, q2, q3, 2 * t * (1 - t))


# Given an initial quaternion q and an initial angular velocity w, find another quaternion such that the SLERP arc
# yields an angular velocity vector parallel to w. Since the length of this arc doesn't affect the direction of this
# angular velocity vector (just the magnitude), c is a constant scaling factor that is tunable
def q_for_w(q, w, c):
    return conjugate(multiply(conjugate(q), exponential_map(c * w)))


# This doesn't work... sad. Try to figure out the actual math behind this curvature condition
def q_for_a(q, w, a, c):
    c = 1 / 5  # hardcode hack
    qw = q_for_w(q, w, c)
    qw2 = q_for_w(qw, a, c)
    return qw2


# Interpolate two rotations, matching boundary conditions on angular velocity
def decasteljau_vel_bcs(q0, qf, w0, wf, t):
    q1 = q_for_w(q0, -w0, 1 / 3)
    q2 = q_for_w(qf, wf, 1 / 3)
    return cubic_decasteljau_slerp(q0, q1, q2, qf, t)


# Interpolate two rotations, matching boundary conditions on angular velocity, and set the angular acceleration at
# either end to be 0
def decasteljau_vel_0_accel_bc(q0, qf, w0, wf, t):
    # NOTE:
    # I tried setting the control points for arbitrary alpha vectors but could not get this to work properly
    # Instead, the control points are placed so that the angular aceleration at either end is 0
    q1 = q_for_w(q0, -w0, 1 / 5)
    q2 = q_for_w(q0, -w0, 2 / 5)
    q3 = q_for_w(qf, wf, 2 / 5)
    q4 = q_for_w(qf, wf, 1 / 5)
    return quintic_decasteljau_slerp(q0, q1, q2, q3, q4, qf, t)


def _test_vel_bcs():
    # np.random.seed(0)
    q0 = random_quaternion()
    qf = random_quaternion()
    print("Start q: ", q0)
    print("End q: ", qf)
    w0 = np.array([0.1, 0.2, 0.3])
    wf = np.array([0.15, 0.25, 0.35])
    t = np.linspace(0, 1, 500)
    dt = t[1] - t[0]
    quats = decasteljau_vel_bcs(q0, qf, w0, wf, t)
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(None, quats, None, omega, None, alpha, t)
    print("Start w: ", omega[0])
    print("End w: ", omega[-1])
    pybullet.connect(pybullet.GUI)
    animate_rotation(traj.quaternions, 5)
    input("Animation complete, press Enter to exit")
    pybullet.disconnect()
    traj.plot()


def _test_vel_accel_bcs():
    # np.random.seed(0)
    q0 = random_quaternion()
    qf = random_quaternion()
    print("Start q: ", q0)
    print("End q: ", qf)
    w0 = np.array([0.1, 0.2, 0.3])
    wf = np.array([0.15, 0.25, 0.35])
    t = np.linspace(0, 1, 500)
    dt = t[1] - t[0]
    quats = decasteljau_vel_0_accel_bc(q0, qf, w0, wf, t)
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(None, quats, None, omega, None, alpha, t)
    print("Start w: ", omega[0])
    print("End w: ", omega[-1])
    print("Start a: ", alpha[0])
    print("End a: ", alpha[-1])
    pybullet.connect(pybullet.GUI)
    animate_rotation(traj.quaternions, 5)
    input("Animation complete, press Enter to exit")
    pybullet.disconnect()
    traj.plot()


if __name__ == "__main__":
    _test_vel_bcs()
    _test_vel_accel_bcs()
