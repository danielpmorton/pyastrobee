"""

TODO
- Check out the SQUAD algorithm. This seems very similar to a cubic decasteljau spherical bezier curve but with a
  different parameterization of the "t" interpolation value
"""

import time
from typing import Union, Optional

import pybullet
import cvxpy as cp
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from pyastrobee.utils.quaternion_class import Quaternion
from pyastrobee.utils.quaternions import (
    quaternion_slerp,
    quaternion_dist,
    random_quaternion,
    quats_to_angular_velocities,
    quaternion_angular_error,
    conjugate,
)
from pyastrobee.trajectories.bezier import BezierCurve, plot_1d_bezier_curve
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.debug_visualizer import visualize_quaternion


from pytransform3d.rotations import (
    compact_axis_angle_from_quaternion,
    concatenate_quaternions,
    quaternion_from_compact_axis_angle,
)
from pytransform3d.rotations._quaternion_operations import concatenate_quaternions


def exponential_map(w):
    q = Quaternion(wxyz=quaternion_from_compact_axis_angle(w))
    return q.xyzw


def log_map(q):
    q = Quaternion(xyzw=q)
    return compact_axis_angle_from_quaternion(q.wxyz)


def pure(w):
    return np.array([*w, 0])


def multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def multiply_2(q1, q2):
    q1 = Quaternion(xyzw=q1)
    q2 = Quaternion(xyzw=q2)
    q3 = Quaternion(wxyz=concatenate_quaternions(q1.wxyz, q2.wxyz))
    return q3.xyzw


def cubic_decasteljau_slerp(q0, q1, q2, q3, t):
    # Based on splines python package
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


def squad(q0, q1, q2, q3, t):
    return cubic_decasteljau_slerp(q0, q1, q2, q3, 2 * t * (1 - t))


def quintic_decasteljau_slerp(q0, q1, q2, q3, q4, q5, t):
    # Based on splines python package
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


def q_for_w(q, w, c):
    return conjugate(multiply(conjugate(q), exponential_map(c * w)))


def _random_testing():
    q1 = random_quaternion()
    q2 = random_quaternion()
    q3 = multiply(q1, q2)
    q4 = multiply_2(q1, q2)
    print(q3)
    print(q4)
    w = np.array([0.1, 0.2, 0.3])
    q = exponential_map(w)
    print(q)
    w = log_map(q)
    print(w)
    q2 = q_for_w(q, w, 1)
    w2 = log_map(multiply(q, conjugate(q2)))
    print(w2)


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
    # traj.plot()
    animate(traj.quaternions)


def animate(quats):
    pybullet.connect(pybullet.GUI)
    cube = create_box((0, 0, 0), (0, 0, 0, 1), 1, (1, 1, 1), True)
    sleep_time = 5 / quats.shape[0]  # 5 second animation
    for quat in quats:
        # visualize_quaternion(quat, lifetime=5 * sleep_time)
        pybullet.resetBasePositionAndOrientation(cube, (0, 0, 0), quat)
        pybullet.stepSimulation()
        time.sleep(sleep_time)
    input("Press Enter to close pybullet")
    pybullet.disconnect()


def decasteljau_vel_bcs(q0, qf, w0, wf, t):
    q1 = q_for_w(q0, -w0, 1 / 3)
    q2 = q_for_w(qf, wf, 1 / 3)
    return cubic_decasteljau_slerp(q0, q1, q2, qf, t)


def q_for_a(q, w, a, c):
    qw = q_for_w(q, w, 1 / 5)  # not needed?
    wnew = w + c * a  # (1 / 5) * a
    qw2 = q_for_w(qw, w, 2 / 5)
    return qw2


def decasteljau_vel_0_accel_bc(q0, qf, w0, wf, t, a0=None, af=None):
    # NOTE:
    # I tried setting the control points for arbitrary alpha vectors but could not get this to work properly
    # Instead, the control points are placed so that the angular aceleration at either end is 0
    q1 = q_for_w(q0, -w0, 1 / 5)
    q2 = q_for_w(q0, -w0, 2 / 5)
    q3 = q_for_w(qf, wf, 2 / 5)
    q4 = q_for_w(qf, wf, 1 / 5)

    if a0 is not None:
        q2 = q_for_a(q0, w0, -a0, 1 / 500)
    if af is not None:
        q3 = q_for_a(qf, wf, af, 1 / 500)  # dt ??

    return quintic_decasteljau_slerp(q0, q1, q2, q3, q4, qf, t)


def _test_vel_accel_bcs():
    # np.random.seed(0)
    q0 = random_quaternion()
    qf = random_quaternion()
    print("Start q: ", q0)
    print("End q: ", qf)
    w0 = np.array([0.1, 0.2, 0.3])
    wf = np.array([0.15, 0.25, 0.35])

    # w0 = np.zeros(3)
    # wf = np.zeros(3)

    t = np.linspace(0, 1, 500)
    dt = t[1] - t[0]
    # quats = decasteljau_vel_0_accel_bc(q0, qf, w0, wf, t)
    a0 = np.array([0.1, 0.1, 0.1])
    af = np.array([0.2, 0.2, 0.2])
    quats = decasteljau_vel_0_accel_bc(q0, qf, w0, wf, t, a0, af)
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(None, quats, None, omega, None, alpha, t)
    print("Start w: ", omega[0])
    print("End w: ", omega[-1])
    print("Start a: ", alpha[0])
    print("End a: ", alpha[-1])
    # traj.plot()
    # animate(traj.quaternions)
    # traj.plot()


def _test_just_slerp():
    # np.random.seed(0)
    q0 = random_quaternion()
    qf = random_quaternion()
    print("Start q: ", q0)
    print("End q: ", qf)
    t = np.linspace(0, 1, 500)
    dt = t[1] - t[0]
    quats = quaternion_slerp(q0, qf, t)
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(None, quats, None, omega, None, alpha, t)
    print("Start w: ", omega[0])
    print("End w: ", omega[-1])
    print("Start a: ", alpha[0])
    print("End a: ", alpha[-1])
    # traj.plot()
    animate(traj.quaternions)
    traj.plot()


def _test_squad():
    # np.random.seed(0)
    q0 = random_quaternion()
    qf = random_quaternion()
    print("Start q: ", q0)
    print("End q: ", qf)
    w0 = np.array([0.1, 0.2, 0.3])
    wf = np.array([0.15, 0.25, 0.35])
    t = np.linspace(0, 1, 500)
    dt = t[1] - t[0]
    q1 = q_for_w(q0, -w0, 1 / 3)
    q2 = q_for_w(qf, wf, 1 / 3)
    quats = squad(q0, q1, q2, qf, t)
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(None, quats, None, omega, None, alpha, t)
    print("Start w: ", omega[0])
    print("End w: ", omega[-1])
    print("Start a: ", alpha[0])
    print("End a: ", alpha[-1])
    # traj.plot()
    animate(traj.quaternions)
    traj.plot()


if __name__ == "__main__":
    # _test_vel_bcs()
    _test_vel_accel_bcs()
    # _test_just_slerp()
    # _test_squad()
