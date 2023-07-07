"""Trajectory generation methods for the two-robot case"""

import numpy as np

from pyastrobee.trajectories.trajectory import Trajectory
from pyastrobee.utils.transformations import (
    make_transform_mat,
    invert_transform_mat,
    transform_point,
)

from pyastrobee.utils.poses import pos_quat_to_tmat
from pyastrobee.utils.rotations import rmat_to_quat
from pyastrobee.utils.quaternions import quats_to_angular_velocities


def offset_trajectory(
    reference_traj: Trajectory, offset_transform: np.ndarray
) -> Trajectory:
    """Construct a trajectory with a fixed offset to a given reference trajectory

    This can also be thought as "Leader/Follower" where leader has the reference trajectory, and follower has the offset
    trajectory. However, this can generalize to more cases than just leader/follower

    Args:
        reference_traj (Trajectory): Trajectory to use as a reference
        offset_transform (np.ndarray): "Offset to reference" transformation matrix, dictating the positional/angular
            difference from the reference trajectory. Shape (4, 4)

    Returns:
        Trajectory: Trajectory with a fixed offset to the reference trajectory
    """
    # TODO: check that the transformation matrix is valid?

    # Transform each pose according to the offset transform
    offset_positions = np.zeros_like(reference_traj.positions)
    offset_quats = np.zeros_like(reference_traj.quaternions)
    for i in range(reference_traj.num_timesteps):
        pose = reference_traj.poses[i]
        T_R2W = pos_quat_to_tmat(pose)  # Reference to World
        T_O2W = T_R2W @ offset_transform  # Offset to World
        offset_positions[i] = T_O2W[:3, 3]
        offset_quats[i] = rmat_to_quat(T_O2W[:3, :3])

    # Take the gradients of the poses to determine derivative information
    offset_vels = np.gradient(offset_positions, reference_traj.times, axis=0)
    offset_accels = np.gradient(offset_vels, reference_traj.times, axis=0)
    offset_omegas = quats_to_angular_velocities(
        offset_quats, np.gradient(reference_traj.times)
    )
    offset_alphas = np.gradient(offset_omegas, reference_traj.times, axis=0)
    return Trajectory(
        offset_positions,
        offset_quats,
        offset_vels,
        offset_omegas,
        offset_accels,
        offset_alphas,
        reference_traj.times,
    )


def dual_trajectory(
    reference_traj: Trajectory, transform_A: np.ndarray, transform_B: np.ndarray
) -> tuple[Trajectory, Trajectory]:
    """Construct two trajectories moving about a reference trajectory with fixed offset transformations

    This can be used as a slightly better formulation of leader/follower to ensure similar trajectory dynamics
    for two robots moving together about a reference

    Args:
        reference_traj (Trajectory): Trajectory to use as a central reference trajectory
        transform_A (np.ndarray): "Offset to reference" transformation matrix for the first trajectory
        transform_B (np.ndarray): "Offset to reference" transformation matrix for the second trajectory

    Returns:
        tuple[Trajectory, Trajectory]: The two trajectories, both offset about the reference
    """
    return offset_trajectory(reference_traj, transform_A), offset_trajectory(
        reference_traj, transform_B
    )


def _test_dual_trajectory():
    # pylint: disable=import-outside-toplevel
    import time
    import pybullet
    from pyastrobee.core.astrobee import Astrobee
    from pyastrobee.utils.rotations import Rx
    from pyastrobee.trajectories.planner import plan_trajectory
    from pyastrobee.control.force_torque_control import ForceTorqueController
    from pyastrobee.utils.debug_visualizer import visualize_path

    pybullet.connect(pybullet.GUI)
    duration = 5
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = [0, -0.5, 0, 0, 0, 0, 1]
    start_vel_1 = np.zeros(3)
    start_omega_1 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    end_pose_1 = [2, -0.5, 1, *rmat_to_quat(Rx(np.pi))]
    end_vel_1 = np.zeros(3)
    end_omega_1 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_alpha_1 = np.zeros(3)

    reference_traj = plan_trajectory(
        start_pose_1[:3],
        start_pose_1[3:],
        start_vel_1,
        start_omega_1,
        start_accel_1,
        start_alpha_1,
        end_pose_1[:3],
        end_pose_1[3:],
        end_vel_1,
        end_omega_1,
        end_accel_1,
        end_alpha_1,
        duration,
        dt,
    )
    # TODO make the offset distance a parameter to tune
    # (get the correct values depending on the bag)
    # The rotation will probably always be the same unless we mess with the arm joints
    offset_distance = 1
    T_A = make_transform_mat(Rx(np.pi), [0, 0, -1 * offset_distance])
    T_B = make_transform_mat(Rx(0), [0, 0, 1 * offset_distance])

    traj_A, traj_B = dual_trajectory(reference_traj, T_A, T_B)
    robot_A = Astrobee(traj_A.poses[0])
    robot_B = Astrobee(traj_B.poses[0])

    # These need to be better tuned for this example
    # I think some of the tracking error is due to high centrifugal forces in this example
    # which causes some error due to the COM offset from the arm
    kp, kv, kq, kw = 50, 10, 2, 0.2
    leader_controller = ForceTorqueController(
        robot_A.id, robot_A.mass, robot_A.inertia, kp, kv, kq, kw, dt
    )
    follower_controller = ForceTorqueController(
        robot_B.id, robot_B.mass, robot_B.inertia, kp, kv, kq, kw, dt
    )

    n_viz = 20
    traj_A.visualize(n_viz)
    visualize_path(traj_A.positions, n_viz, (1, 1, 1))
    traj_B.visualize(n_viz)
    visualize_path(traj_B.positions, n_viz, (1, 1, 1))
    visualize_path(reference_traj.positions, n_viz, (1, 1, 1))

    for i in range(traj_A.num_timesteps):
        pos_1, orn_1, lin_vel_1, ang_vel_1 = robot_A.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = robot_B.dynamics_state
        leader_controller.step(
            pos_1,
            lin_vel_1,
            orn_1,
            ang_vel_1,
            traj_A.positions[i, :],
            traj_A.linear_velocities[i, :],
            traj_A.linear_accels[i, :],
            traj_A.quaternions[i, :],
            traj_A.angular_velocities[i, :],
            traj_A.angular_accels[i, :],
        )
        follower_controller.step(
            pos_2,
            lin_vel_2,
            orn_2,
            ang_vel_2,
            traj_B.positions[i, :],
            traj_B.linear_velocities[i, :],
            traj_B.linear_accels[i, :],
            traj_B.quaternions[i, :],
            traj_B.angular_velocities[i, :],
            traj_B.angular_accels[i, :],
        )
        time.sleep(1 / 120)

    # Stopping mode
    while True:
        pos_1, orn_1, lin_vel_1, ang_vel_1 = robot_A.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = robot_B.dynamics_state
        leader_controller.step(
            pos_1,
            lin_vel_1,
            orn_1,
            ang_vel_1,
            traj_A.positions[-1, :],
            traj_A.linear_velocities[-1, :],
            traj_A.linear_accels[-1, :],
            traj_A.quaternions[-1, :],
            traj_A.angular_velocities[-1, :],
            traj_A.angular_accels[-1, :],
        )
        follower_controller.step(
            pos_2,
            lin_vel_2,
            orn_2,
            ang_vel_2,
            traj_B.positions[-1, :],
            traj_B.linear_velocities[-1, :],
            traj_B.linear_accels[-1, :],
            traj_B.quaternions[-1, :],
            traj_B.angular_velocities[-1, :],
            traj_B.angular_accels[-1, :],
        )
        time.sleep(1 / 120)


if __name__ == "__main__":
    # _test_leader_follower()
    _test_dual_trajectory()
