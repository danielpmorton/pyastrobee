"""Script for PID trajectory tracking testing / debugging

TODO
- Merge polynomial_trajectory file with something else -- planner?
- Add ability to save the true state information and plot it against the traj
"""

import time

import numpy as np
import pybullet
import pytransform3d.rotations as pr

from pyastrobee.control.trajectory import visualize_traj
from pyastrobee.control.polynomial_trajectories import polynomial_trajectory
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.quaternion import random_quaternion, xyzw_to_wxyz
from pyastrobee.utils.debug_visualizer import visualize_frame, remove_debug_objects
from pyastrobee.utils.poses import pos_quat_to_tmat


def box_inertia(m, l, w, h):
    return (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])


# Remember wesley's comments about the lambda matrices being mainly relevant for operational space control
# for multi-body robot system
# And the I_m0 symbol refers to a 6x6 identity matrix
# Also f* refers to the idealized unit mass system
# So, modifying the equations from the slides to just use the mass
def get_force(m, kv, kp, x, dx, x_des, dx_des, d2x_des):
    return m * d2x_des - kv * (dx - dx_des) - kp * (x - x_des)


def get_torque(I, kw, kq, q, w, q_des, w_des, a_des):
    Q = xyzw_to_wxyz(np.row_stack([q_des, q]))
    ang_err = pr.compact_axis_angle_from_quaternion(
        pr.concatenate_quaternions(Q[1], pr.q_conj(Q[0]))
    )
    return I @ a_des - kw * (w - w_des) - kq * ang_err


def main():
    pybullet.connect(pybullet.GUI)
    np.random.seed(0)
    pose_1 = [0, 0, 0, 0, 0, 0, 1]
    pose_2 = [1, 1, 1, *random_quaternion()]
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
    x_des, y_des, z_des = traj.positions[-1, :]
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
        # wx, wy, wz = ang_vel
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


if __name__ == "__main__":
    main()
