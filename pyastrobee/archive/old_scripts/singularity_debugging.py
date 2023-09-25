"""Script to help debug a rotation control instability

Archived since the problem has been solved

Notes from the debugging process:
- Using getBasePositionAndOrientation in pybullet when following a trajectory can sometimes result in a sign flip in
  the recorded quaternion (due to quaternion double cover)
- The previous version of the controller did not handle sign flips in quaternions, so the angular error vector was
  instantaneously flipped as well, completely throwing off the rotation control
- Updating the controller to handle sign flips fixed this

Other notes:
- Ensure that the quaternion curves are as smooth as possible - instantaneous changes in the quaternion command
  (even if small) can sometimes lead to instability, depending on controller gains. For instance, if "stopping"
  at the end of a trajectory, use the last quaternion from the trajectory, rather than an explicit goal orientation.
  These values might be slightly different, and the angular error vector is best suited to small errors
- Ensure that the kw gain is large enough to dampen some rotational vibrations
"""


import numpy as np
import matplotlib.pyplot as plt
from pyastrobee.core.astrobee import Astrobee
from pyastrobee.control.force_torque_control import ForceTorqueController
from pyastrobee.utils.bullet_utils import initialize_pybullet
from pyastrobee.config.astrobee_motion import MAX_FORCE_MAGNITUDE, MAX_TORQUE_MAGNITUDE
from pyastrobee.trajectories.quaternion_interpolation import (
    quaternion_interpolation_with_bcs,
)
from pyastrobee.utils.quaternions import quats_to_angular_velocities
from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.bullet_utils import create_box
from pyastrobee.utils.dynamics import box_inertia
from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.debug_visualizer import visualize_frame
from pyastrobee.utils.quaternions import quaternion_angular_error


def controller_test(start_pose, end_pose):
    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    end_tmat = pos_quat_to_tmat(end_pose)
    visualize_frame(end_tmat)
    robot = Astrobee(start_pose)
    robot.store_arm(force=True)
    controller = ForceTorqueController(
        robot.id,
        robot.mass,
        robot.inertia,
        20,
        20,
        1,  # 1,  # 0.1,
        1,
        dt,
        # max_force=MAX_FORCE_MAGNITUDE * 10,  # TODO REMOVE
        # max_torque=MAX_TORQUE_MAGNITUDE * 10,
    )
    tf = 10
    n_timesteps = round(tf / dt)
    quats = quaternion_interpolation_with_bcs(
        start_pose[3:],
        end_pose[3:],
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        tf,
        n_timesteps,
    )
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(
        start_pose[:3] * np.ones((n_timesteps, 1)),
        quats,
        np.zeros((n_timesteps, 3)),
        omega,
        np.zeros((n_timesteps, 3)),
        alpha,
        np.arange(n_timesteps) * dt,
    )
    quat_hist = np.zeros((traj.num_timesteps, 4))
    try:
        controller.follow_traj(traj, False)
        # Bonus time (same length as original traj)
        for i in range(traj.num_timesteps):
            pos, orn, vel, omega = controller.get_current_state()
            quat_hist[i] = orn
            controller.step(
                pos,
                vel,
                orn,
                omega,
                traj.positions[-1],
                np.zeros(3),
                np.zeros(3),
                traj.quaternions[-1],
                np.zeros(3),
                np.zeros(3),
            )
    # time.sleep(1 / 120)
    finally:
        print("CONTROLLER QUAT SIGN: ", controller.quat_sign)
        traj.plot()
        plot_quaternion_history(quat_hist, show=False)
        controller.plot_ang_err()
        # traj.plot(False)
        controller.traj_log.plot(show=False)
        controller.control_log.plot(show=False)
        plt.show()


def quaternion_test(start_pose, end_pose):
    """Test to see if just using the quaternions themselves we see any weirdness in the ang error"""
    dt = 1 / 350
    tf = 10
    n_timesteps = round(tf / dt)
    quats = quaternion_interpolation_with_bcs(
        start_pose[3:],
        end_pose[3:],
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        tf,
        n_timesteps,
    )
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(
        start_pose[:3] * np.ones((n_timesteps, 1)),
        quats,
        np.zeros((n_timesteps, 3)),
        omega,
        np.zeros((n_timesteps, 3)),
        alpha,
        np.arange(n_timesteps) * dt,
    )
    n = quats.shape[0]
    errors = np.zeros((n - 1, 3))
    for i in range(n - 1):
        q = quats[i]
        q_des = quats[i + 1]
        errors[i] = quaternion_angular_error(q, q_des)

    plot_angular_errors(errors)


def reset_test(start_pose, end_pose):
    """Test to see if resetting the base position and orientation will affect the results"""

    client = initialize_pybullet()
    dt = client.getPhysicsEngineParameters()["fixedTimeStep"]
    end_tmat = pos_quat_to_tmat(end_pose)
    visualize_frame(end_tmat)
    robot = Astrobee(start_pose)
    robot.store_arm(force=True)

    tf = 10
    n_timesteps = round(tf / dt)
    quats = quaternion_interpolation_with_bcs(
        start_pose[3:],
        end_pose[3:],
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        tf,
        n_timesteps,
    )
    omega = quats_to_angular_velocities(quats, dt)
    alpha = np.gradient(omega, dt, axis=0)
    traj = Trajectory(
        start_pose[:3] * np.ones((n_timesteps, 1)),
        quats,
        np.zeros((n_timesteps, 3)),
        omega,
        np.zeros((n_timesteps, 3)),
        alpha,
        np.arange(n_timesteps) * dt,
    )
    n = quats.shape[0]
    errors = np.zeros((n - 1, 3))
    quat_hist = np.zeros((n - 1, 4))
    for i in range(n - 1):
        client.resetBasePositionAndOrientation(robot.id, np.zeros(3), quats[i])
        q = client.getBasePositionAndOrientation(robot.id)[1]
        quat_hist[i] = q
        q_des = quats[i + 1]
        errors[i] = quaternion_angular_error(q, q_des)

    plot_angular_errors(errors)
    plot_quaternion_history(quat_hist)


def plot_angular_errors(errors):
    errors = np.atleast_2d(errors)
    n = errors.shape[0]
    x = np.arange(n)
    plt.figure()
    plt.plot(x, np.linalg.norm(errors, axis=1))
    plt.title("Angular error vector magnitude")
    plt.figure()
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(x, errors[:, i])
        plt.ylim
    plt.suptitle("Angular error vector components")
    # plt.show()


def plot_quaternion_history(quats, show=False):
    n = quats.shape[0]
    x = np.arange(n)
    plt.figure()
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.plot(x, quats[:, i])
    plt.suptitle("Recorded quaternion components")
    if show:
        plt.show()


if __name__ == "__main__":
    # Note for reference: these work great (These are just two random quaternions that I liked)
    # fmt: off
    start_pose = [0, 0, 0, 0.47092437141245563, 0.8134303596981151, 0.0001291583857834318, 0.3414107052353368]
    end_pose = [0, 0, 0, 0.34196961983991075, 0.21516679142364756, 0.4340223300167253, 0.8052233528790954]
    # fmt: on

    # These are problematic
    start_pose = [0, 0, 0, 0, 0, 1, 0]
    end_pose = [0, 0, 0, 0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2]

    controller_test(start_pose, end_pose)
    # quaternion_test(start_pose, end_pose)
    # reset_test(start_pose, end_pose)
    plt.show()
