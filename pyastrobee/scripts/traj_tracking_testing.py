"""Script for PID trajectory tracking testing / debugging

TODO
- Merge polynomial_trajectory file with something else -- planner?
- Add ability to save the true state information and plot it against the traj !!!! PLOT !!!
- Move quaternion dist to the quaternion file
"""

import time
from typing import Union

import numpy as np
import numpy.typing as npt
import pybullet
import pytransform3d.rotations as pr

from pyastrobee.control.trajectory import (
    visualize_traj,
    Trajectory,
    compare_trajs,
    stopping_criteria,
)
from pyastrobee.control.polynomial_trajectories import polynomial_trajectory
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.quaternion_class import Quaternion
from pyastrobee.utils.quaternion import (
    random_quaternion,
    xyzw_to_wxyz,
    conjugate,
    combine_quaternions,
    check_quaternion,
)
from pyastrobee.utils.rotations import quat_to_fixed_xyz


def box_inertia(m, l, w, h):
    return (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])


# Remember wesley's comments about the lambda matrices being mainly relevant for operational space control
# for multi-body robot system
# And the I_m0 symbol refers to a 6x6 identity matrix
# Also f* refers to the idealized unit mass system
# So, modifying the equations from the slides to just use the mass
def get_force(m, kv, kp, x, dx, x_des, dx_des, d2x_des):
    return m * d2x_des - kv * (dx - dx_des) - kp * (x - x_des)


def get_force_new(
    mass: float,
    kv: float,
    kp: float,
    cur_pos: npt.ArrayLike,
    cur_vel: npt.ArrayLike,
    des_pos: npt.ArrayLike,
    des_vel: npt.ArrayLike,
    des_accel: npt.ArrayLike,
) -> np.ndarray:
    M = mass * np.eye(3)
    pos_err = np.subtract(cur_pos, des_pos)
    vel_err = np.subtract(cur_vel, des_vel)
    return M @ np.asarray(des_accel) - kv * vel_err - kp * pos_err


def get_torque(I, kw, kq, q, w, q_des, w_des, a_des):
    Q = xyzw_to_wxyz(np.row_stack([q_des, q]))
    ang_err_old = pr.compact_axis_angle_from_quaternion(
        pr.concatenate_quaternions(Q[1], pr.q_conj(Q[0]))
    )
    # NEW TESTING - NOT DONE
    q_diff = combine_quaternions(q, conjugate(q_des))
    ang_err = quat_to_fixed_xyz(q_diff)
    print("Old: ", ang_err_old)
    print("New: ", ang_err)
    return I @ a_des - kw * (w - w_des) - kq * ang_err


# NOT DONE
def get_torque_new(inertia, kw, kq, cur_q, cur_w, des_q, des_w, des_a):
    pass


def main():
    tracker = Trajectory()
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
    point = np.array([0.0, 0.0, 0.0])  # Point in base frame where force is applied
    inertia = box_inertia(mass, *sidelengths)
    for i, t in enumerate(traj.times):
        # fid = visualize_frame(pos_quat_to_tmat(traj_poses[i]))
        pos, q = pybullet.getBasePositionAndOrientation(box)
        x, y, z = pos
        lin_vel, ang_vel = pybullet.getBaseVelocity(box)
        tracker.log_state(pos, q, lin_vel, ang_vel, dt)
        vx, vy, vz = lin_vel
        wx, wy, wz = ang_vel
        x_des, y_des, z_des = traj.positions[i, :]
        vx_des, vy_des, vz_des = traj.linear_velocities[i, :]
        ax_des, ay_des, az_des = traj.linear_accels[i, :]
        q_des = traj.quaternions[i, :]
        omega_des = traj.angular_velocities[i, :]
        alpha_des = traj.angular_accels[i, :]
        Fx = get_force(mass, kv, kp, x, vx, x_des, vx_des, ax_des)
        Fy = get_force(mass, kv, kp, y, vy, y_des, vy_des, ay_des)
        Fz = get_force(mass, kv, kp, z, vz, z_des, vz_des, az_des)
        tau = get_torque(inertia, kw, kq, q, ang_vel, q_des, omega_des, alpha_des)
        force = np.array([Fx, Fy, Fz])
        pybullet.applyExternalForce(
            box, base_idx, list(force), list(pos), pybullet.WORLD_FRAME
        )
        pybullet.applyExternalTorque(box, base_idx, list(tau), pybullet.WORLD_FRAME)
        pybullet.stepSimulation()
        time.sleep(dt)
    print("Trajectory complete. Stopping...")
    pos_des = traj.positions[-1, :]
    x_des, y_des, z_des = pos_des
    q_des = traj.quaternions[-1, :]
    vx_des, vy_des, vz_des = [0.0, 0.0, 0.0]
    ax_des, ay_des, az_des = [0.0, 0.0, 0.0]
    omega_des = np.array([0.0, 0.0, 0.0])
    alpha_des = np.array([0.0, 0.0, 0.0])
    while True:
        pos, q = pybullet.getBasePositionAndOrientation(box)
        x, y, z = pos
        lin_vel, ang_vel = pybullet.getBaseVelocity(box)
        vx, vy, vz = lin_vel
        wx, wy, wz = ang_vel
        if stopping_criteria(pos, q, lin_vel, ang_vel, pos_des, q_des):
            break
        Fx = get_force(mass, kv, kp, x, vx, x_des, vx_des, ax_des)
        Fy = get_force(mass, kv, kp, y, vy, y_des, vy_des, ay_des)
        Fz = get_force(mass, kv, kp, z, vz, z_des, vz_des, az_des)
        force = np.array([Fx, Fy, Fz])
        pybullet.applyExternalForce(
            box, base_idx, force, list(pos), pybullet.WORLD_FRAME
        )
        tau = get_torque(inertia, kw, kq, q, ang_vel, q_des, omega_des, alpha_des)
        pybullet.applyExternalTorque(box, base_idx, list(tau), pybullet.WORLD_FRAME)
        pybullet.stepSimulation()
        time.sleep(dt)

    pybullet.disconnect()
    compare_trajs(traj, tracker)


# THIS NEEDS UPDATING
# AND MOVE THIS TO QUATERNIONS FILE
def quaternion_angular_difference(
    q1: Union[Quaternion, npt.ArrayLike], q2: Union[Quaternion, npt.ArrayLike]
):
    """Gives a world-frame compact-axis-angle form of the angular error between two quaternions (q1 -> q2)

    - This is similar (but not the same) as a difference between fixed-XYZ conventions. Differences between Euler/Fixed
        angle sets do not often represent true rotations

    Args:
        q1 (Union[Quaternion, npt.ArrayLike]): _description_
        q2 (Union[Quaternion, npt.ArrayLike]): _description_

    Returns:
        _type_: _description_
    """
    # Need to work with quaterions in WXYZ for for pytransform3d
    if isinstance(q1, Quaternion):
        q1 = q1.wxyz
    else:
        q1 = check_quaternion(q1)
        q1 = np.ravel(xyzw_to_wxyz(q1))
    if isinstance(q2, Quaternion):
        q2 = q2.wxyz
    else:
        q2 = check_quaternion(q2)
        q2 = np.ravel(xyzw_to_wxyz(q2))
    # This line is based on the math from pytransform3d's quaternion_gradient(),
    # but simplified to work with just 2 quats and no time info
    return pr.compact_axis_angle_from_quaternion(
        pr.concatenate_quaternions(q2, pr.q_conj(q1))
    )


# NEEDS WORK AND TESTING
def quaternion_diff(q1, q2):
    # Gives the quaternion representing the rotation from q1 -> q2
    return combine_quaternions(q1, conjugate(q2))


if __name__ == "__main__":
    main()
