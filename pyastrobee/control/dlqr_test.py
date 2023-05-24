"""
TODO
This isn't exactly working right now - it says that there is no finite solution
I think this is probably because I formulated this based on the dynamics of the robot 
rather than the error dynamics? But, then we might expect that the robot would just be driven
directly to [0, 0, 0, 0, 0, 0, 0]? Which doesn't make sense by itself because the quaternion is invalid

"""


import time

import pybullet
import numpy as np
import control

from pyastrobee.trajectories.trajectory import stopping_criteria, visualize_traj
from pyastrobee.trajectories.polynomial_trajectories import polynomial_trajectory
from pyastrobee.utils.quaternions import random_quaternion
from pyastrobee.utils.bullet_utils import create_box


# Helper function to get the moment of inertia of our debugging box
def box_inertia(m, l, w, h):
    return (1 / 12) * m * np.diag([w**2 + h**2, l**2 + h**2, l**2 + w**2])


# Matrix for relating the angular velocity to the quaternion derivative
# Check the math here? Somewhat confident that it's correct
def make_omega(q):
    x, y, z, w = q
    return (1 / 2) * np.array([[w, z, y], [z, w, -x], [-y, x, w], [-x, y, -z]])


# Linearized dynamics matrix
def get_A(q, dt):
    return np.block(
        [
            [np.eye(3), np.zeros((3, 4)), dt * np.eye(3), np.zeros((3, 3))],
            [np.zeros((4, 3)), np.eye(4), np.zeros((4, 3)), dt * make_omega(q)],
            [np.zeros((3, 3)), np.zeros((3, 4)), np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 4)), np.zeros((3, 3)), np.eye(3)],
        ]
    )


# Linearized control matrix
def get_B(m, inertia, q, dt):
    # NOTE this IGNORES the centripetal terms in the torque/acceleration relationship
    # (because this is a feedthrough term that doesn't quite work for this situation)
    # TODO see if there is a way to get this to work - add extra columns to the matrix?
    # ACTUALLY should probs just be added on to the control value u!! -- check this
    inertia_inv = np.linalg.inv(inertia)
    return np.block(
        [
            [dt**2 / (2 * m) * np.eye(3), np.zeros((3, 3))],
            [np.zeros((4, 3)), (dt**2 / 2) * make_omega(q) @ inertia_inv],
            [(dt / m) * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), dt * inertia_inv],
        ]
    )


# Let state = [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
def get_state(robot_id):
    pos, quat = pybullet.getBasePositionAndOrientation(robot_id)
    lin_vel, ang_vel = pybullet.getBaseVelocity(robot_id)
    # These are originally tuples, so convert to numpy
    return np.concatenate([pos, quat, lin_vel, ang_vel])


# Calculate the control input = let u = [Fx, Fy, Fz, Tx, Ty, Tz]
def get_u(X_cur, X_des, Q, R, dt, mass, inertia):
    cur_q = X_cur[3:7]
    A = get_A(cur_q, dt)
    B = get_B(mass, inertia, cur_q, dt)
    K, S, E = control.dlqr(A, B, Q, R)
    return -K @ (X_cur - X_des)


# Applies forces and steps the simulation
# (Should control also be calculated here rather than outside?)
def step(robot_id, dt, pos, u):
    # Apply forces and step sim
    # Also calculate the control in this function???
    force = u[:3], torque = u[3:]
    # TODO: explain -1? And does pos need to be a list?
    pybullet.applyExternalForce(robot_id, -1, force, list(pos), pybullet.WORLD_FRAME)
    pybullet.applyExternalTorque(robot_id, -1, list(torque), pybullet.WORLD_FRAME)
    pybullet.stepSimulation()
    time.sleep(dt)


def main():
    pybullet.connect(pybullet.GUI)
    np.random.seed(0)
    pose_1 = [0, 0, 0, 0, 0, 0, 1]
    pose_2 = [1, 2, 3, *random_quaternion()]
    mass = 10
    sidelengths = [0.25, 0.25, 0.25]
    box = create_box(pose_1[:3], pose_1[3:], mass, sidelengths, True)
    max_time = 10
    dt = 0.01
    inertia = box_inertia(mass, *sidelengths)
    traj = polynomial_trajectory(pose_1, pose_2, max_time, dt)
    visualize_traj(traj, 20)

    # Make the Q and R matrices
    # Tune the multiplicative constants as needed
    Q_pos = 1 * np.ones(3)
    Q_orn = 1 * np.ones(4)
    Q_vel = 1 * np.ones(3)
    Q_omega = 1 * np.ones(3)
    Q = np.diag(np.concatenate([Q_pos, Q_orn, Q_vel, Q_omega]))
    R_force = 1 * np.ones(3)
    R_torque = 1 * np.ones(3)
    R = np.diag(np.concatenate([R_force, R_torque]))

    # Trajectory tracking loop
    for i in range(traj.num_timesteps):
        X_des = np.array(
            [
                *traj.positions[i, :],
                *traj.quaternions[i, :],
                *traj.linear_velocities[i, :],
                *traj.angular_velocities[i, :],
            ]
        )
        X = get_state(box)
        # Move the get_u into the step() function???
        u = get_u(X, X_des, Q, R, dt, mass, inertia)
        step(box, dt, X[:3], u)
    # Trajectory complete, switch to stop mode
    des_pos = traj.positions[-1, :]
    des_quat = traj.quaternions[-1, :]
    X_des = np.array([*des_pos, *des_quat, *np.zeros(6)])
    while True:
        X = get_state(box)
        if stopping_criteria(X[:3], X[3:7], X[7:10], X[10:], des_pos, des_quat):
            break
        u = get_u(X, X_des, Q, R, dt, mass, inertia)
        step(box, dt, X[:3], u)


if __name__ == "__main__":
    main()
