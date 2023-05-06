"""Preliminary test to see if we can get operational space control working on the Astrobee

Most of these equations will be based on the CS327A course notes

Major differences from CS327A:
- We're in space, so we don't need to do any gravity compensation (ignore the gravity vector)
- The Astrobee is a floating robot so the definition of the DOFs is a bit different
"""
import time

import pybullet
import numpy as np
from scipy.linalg import block_diag

from pyastrobee.elements.astrobee import Astrobee

# Found this based on a permutation of the array from the textbook
# It seems to differ from the arrays I made the other day though
def lambda_matrix(q):
    x, y, z, w = q
    return np.array([[w, z, -y], [-z, w, x], [y, -x, w], [-x, -y, -z]])


def Er(q):
    return 0.5 * lambda_matrix(q)


def Er_plus(q):
    return 2 * lambda_matrix(q).T


def E(Ep, Er):
    return block_diag(Ep, Er)


def get_current_control_point(t):
    # We'll just use the trajectory from the CS327A HW as a starting point
    # This will just move the end-effector in a "rainbow" motion in y and z while leaving x constant
    tptd5 = 2.0 * np.pi * t / 5.0
    fptd5 = 4.0 * np.pi * t / 5.0
    cos_tptd5 = np.cos(tptd5)
    sin_tptd5 = np.sin(tptd5)
    pos_des = np.array([0, 0.5 + 0.1 * cos_tptd5, 0.65 - 0.05 * np.cos(fptd5)])
    vel_des = np.array([0, -(np.pi / 25.0) * sin_tptd5, (np.pi / 25.0) * np.sin(fptd5)])
    accel_des = np.array(
        [
            0,
            -2.0 * (np.pi**2 / 125.0) * cos_tptd5,
            4.0 * (np.pi**2 / 125.0) * np.cos(fptd5),
        ]
    )
    # The quaternion trajectory will be modified for WXYZ -> XYZW
    qwd = (1 / np.sqrt(2)) * np.sin((np.pi / 4) * cos_tptd5)
    qxd = (1 / np.sqrt(2)) * np.cos((np.pi / 4) * cos_tptd5)
    qyd = qwd
    qzd = qxd
    dqwd = (
        (-np.pi**2 / (10.0 * np.sqrt(2)))
        * np.cos((np.pi / 4) * cos_tptd5)
        * sin_tptd5
    )
    dqxd = (
        (np.pi**2 / (10.0 * np.sqrt(2))) * np.sin((np.pi / 4) * cos_tptd5) * sin_tptd5
    )
    dqyd = dqwd
    dqzd = dqxd
    d2qwd = (-np.pi**4 / (100.0 * np.sqrt(2))) * sin_tptd5**2 * np.sin(
        (np.pi / 4) * cos_tptd5
    ) - (np.pi**3 / (25.0 * np.sqrt(2))) * np.cos((np.pi / 4) * cos_tptd5) * cos_tptd5
    d2qxd = (-np.pi**4 / (100.0 * np.sqrt(2))) * sin_tptd5**2 * np.cos(
        (np.pi / 4) * cos_tptd5
    ) + (np.pi**3 / (25.0 * np.sqrt(2))) * np.sin((np.pi / 4) * cos_tptd5) * cos_tptd5
    d2qyd = d2qwd
    d2qzd = d2qxd
    q_des = np.array([qxd, qyd, qzd, qwd])
    dq_des = np.array([dqxd, dqyd, dqzd, dqwd])
    d2q_des = np.array([d2qxd, d2qyd, d2qzd, d2qwd])
    return pos_des, vel_des, accel_des, q_des, dq_des, d2q_des


def main():
    pybullet.connect(pybullet.GUI)
    robot = Astrobee()
    dt = 1 / 240
    counter = 0
    # Set gains
    op_kp = 50
    op_kv = 10
    joint_damping_kv = 20
    start_time = time.time()
    while True:
        curr_time = time.time() - start_time
        # g = NotImplemented  # Gravity vector (= 0 since we're in space)
        # Get current robot state information
        vel = robot.velocity
        omega = robot.angular_velocity
        pos = robot.position
        orn = robot.orientation
        # Get operational space matrices
        Jv, Jw = robot.get_jacobians(2, [-0.08240211, 0, -0.04971851])  # CHECK THIS
        ndof = Jv.shape[1]
        J0 = np.row_stack([Jv, Jw])
        A = robot.mass_matrix
        A_inv = np.linalg.inv(A)
        L0 = np.linalg.inv(J0 @ A_inv @ J0.T)  # Lambda matrix (get real name)
        Jbar = A_inv @ J0.T @ L0  # Dynamically consistent generalized Jacobian inverse
        N_bar = np.eye(ndof) - Jbar @ J0  # Nullspace projection matrix
        # Get trajectory values
        pos_des, vel_des, accel_des, q_des, dq_des, d2q_des = get_current_control_point(
            curr_time
        )
        # Get the desired angular velocity / acceleration
        omega_des = Er_plus(q_des) @ dq_des
        alpha_des = 2 * lambda_matrix(q_des).T @ d2q_des
        # Compute errors
        q_err = orn - q_des
        angular_err = Er_plus(orn) @ q_err
        pos_err = pos - pos_des
        vel_err = vel - vel_des
        omega_err = omega - omega_des
        # Compute joint torques
        F = accel_des - op_kp * pos_err - op_kv * vel_err
        M = alpha_des - op_kp * angular_err - op_kv * omega_err
        F0_star = np.concatenate([F, M])
        # p = Jbar.T @ g GRAVITY IS 0
        F0 = L0 @ F0_star
        # TODO CHECK ON qdot's JOINT CORRESPONDENCE WITH THE JACOBIAN
        qdot = np.concatenate(
            [robot.velocity, robot.angular_velocity, robot.joint_vels[1:]]
        )
        joint_damping = N_bar.T @ (A @ (-joint_damping_kv * qdot))
        # Compute the torques
        command_torques = joint_damping + J0.T @ F0
        # Command the robot using the torques
        # TODO check on the size of the command torques
        # TODO we might want to eliminate a bunch of columns from the matrices
        # since we don't care about a lot of the joints (like the gripper)

        external_forces = command_torques[:3]
        external_torques = command_torques[3:6]
        joint_torques = command_torques[6:]
        # TODO need to decide if these forces/torques are defined in world or link frame
        # Probably need to evalueate if the first 6 cols of the jacobian change with a rotation of the robot
        pybullet.applyExternalForce(
            robot.id,
            -1,
            external_forces,
            list(pos),
            pybullet.LINK_FRAME,  # LINK OR WORLD??
        )
        pybullet.applyExternalTorque(
            robot.id, -1, list(external_torques), pybullet.LINK_FRAME  # LINK OR WORLD??
        )
        # TODO check on if the arg is "force" or "forces"
        pybullet.setJointMotorControlArray(
            robot.id,
            list(range(1, Astrobee.NUM_JOINTS)), # Ignore 1st fixed joint??
            pybullet.TORQUE_CONTROL,
            forces=list(joint_torques),
        )
        # TODO !!! NEED TO APPLY THE JOINT TORQUES
        # Zero out the torques for the gripper???? Or will they already be zeroed out if we use the jacobian
        # for the arm distal link?
        pybullet.stepSimulation()
        time.sleep(dt)
        counter += 1


if __name__ == "__main__":
    main()
