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
from pyastrobee.utils.quaternions import random_quaternion, quaternion_angular_diff


def box_inertia(m, l, w, h):
    return (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])


# Remember wesley's comments about the lambda matrices being mainly relevant for operational space control
# for multi-body robot system
# And the I_m0 symbol refers to a 6x6 identity matrix
# Also f* refers to the idealized unit mass system


def get_force(
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


def get_torque(
    inertia: npt.ArrayLike,
    kw: float,
    kq: float,
    cur_q: npt.ArrayLike,
    cur_w: npt.ArrayLike,
    des_q: npt.ArrayLike,
    des_w: npt.ArrayLike,
    des_a: npt.ArrayLike,
) -> np.ndarray:
    ang_err = quaternion_angular_diff(des_q, cur_q)
    ang_vel_err = cur_w - des_w
    return inertia @ des_a - kw * ang_vel_err - kq * ang_err


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
    inertia = box_inertia(mass, *sidelengths)
    for i, t in enumerate(traj.times):
        cur_pos, cur_quat = pybullet.getBasePositionAndOrientation(box)
        cur_lin_vel, cur_ang_vel = pybullet.getBaseVelocity(box)
        tracker.log_state(cur_pos, cur_quat, cur_lin_vel, cur_ang_vel, dt)
        des_pos = traj.positions[i, :]
        des_vel = traj.linear_velocities[i, :]
        des_accel = traj.linear_accels[i, :]
        des_quat = traj.quaternions[i, :]
        des_omega = traj.angular_velocities[i, :]
        des_alpha = traj.angular_accels[i, :]
        force = get_force(
            mass, kv, kp, cur_pos, cur_lin_vel, des_pos, des_vel, des_accel
        )
        tau = get_torque(
            inertia, kw, kq, cur_quat, cur_ang_vel, des_quat, des_omega, des_alpha
        )
        pybullet.applyExternalForce(
            box, base_idx, list(force), list(cur_pos), pybullet.WORLD_FRAME
        )
        pybullet.applyExternalTorque(box, base_idx, list(tau), pybullet.WORLD_FRAME)
        pybullet.stepSimulation()
        time.sleep(dt)
    print("Trajectory complete. Stopping...")
    pos_des = traj.positions[-1, :]
    des_quat = traj.quaternions[-1, :]
    des_vel = np.zeros(3)
    des_accel = np.zeros(3)
    des_omega = np.zeros(3)
    des_alpha = np.zeros(3)
    while True:
        cur_pos, cur_quat = pybullet.getBasePositionAndOrientation(box)
        cur_lin_vel, cur_ang_vel = pybullet.getBaseVelocity(box)
        if stopping_criteria(
            cur_pos, cur_quat, cur_lin_vel, cur_ang_vel, pos_des, des_quat
        ):
            break
        force = get_force(
            mass, kv, kp, cur_pos, cur_lin_vel, pos_des, des_vel, des_accel
        )
        tau = get_torque(
            inertia, kw, kq, cur_quat, cur_ang_vel, des_quat, des_omega, des_alpha
        )
        pybullet.applyExternalForce(
            box, base_idx, force, list(cur_pos), pybullet.WORLD_FRAME
        )
        pybullet.applyExternalTorque(box, base_idx, list(tau), pybullet.WORLD_FRAME)
        pybullet.stepSimulation()
        time.sleep(dt)

    pybullet.disconnect()
    compare_trajs(traj, tracker)


if __name__ == "__main__":
    main()
