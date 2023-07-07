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


def follower_trajectory(leader_traj: Trajectory, T_F2L: np.ndarray) -> Trajectory:
    follower_positions = np.zeros_like(leader_traj.positions)
    follower_quats = np.zeros_like(leader_traj.quaternions)
    # follower_vels = np.zeros_like(leader_traj.linear_velocities)
    # follower_omegas = np.zeros_like(leader_traj.angular_velocities)
    # follower_accels = np.zeros_like(leader_traj.linear_accels)
    # follower_alphas = np.zeros_like(leader_traj.angular_accels)

    # Assume we have a fixed timestep for now (TODO?)
    dt = leader_traj.times[1] - leader_traj.times[0]

    #
    for i in range(leader_traj.num_timesteps):
        pose = leader_traj.poses[i]
        T_L2W = pos_quat_to_tmat(pose)
        T_F2W = T_L2W @ T_F2L
        follower_positions[i] = T_F2W[:3, 3]
        follower_quats[i] = rmat_to_quat(T_F2W[:3, :3])

    follower_vels = np.gradient(follower_positions, dt, axis=0)
    follower_accels = np.gradient(follower_vels, dt, axis=0)
    follower_omegas = quats_to_angular_velocities(follower_quats, dt)
    follower_alphas = np.gradient(follower_omegas, dt, axis=0)
    return Trajectory(
        follower_positions,
        follower_quats,
        follower_vels,
        follower_omegas,
        follower_accels,
        follower_alphas,
        leader_traj.times,
    )


def _test_leader_follower():
    # pylint: disable=import-outside-toplevel
    import time
    import pybullet
    from pyastrobee.core.astrobee import Astrobee
    from pyastrobee.utils.rotations import Rx
    from pyastrobee.trajectories.planner import plan_trajectory
    from pyastrobee.control.force_torque_control import ForceTorqueController

    pybullet.connect(pybullet.GUI)
    duration = 5
    dt = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
    start_pose_1 = [0, -0.5, 0, *rmat_to_quat(Rx(np.pi / 2))]
    start_vel_1 = [0.3, 0.3, 0.3]
    start_omega_1 = np.zeros(3)
    start_accel_1 = np.zeros(3)
    start_alpha_1 = np.zeros(3)
    end_pose_1 = [2, -0.5, 1, *rmat_to_quat(Rx(-np.pi / 2))]
    end_vel_1 = start_vel_1
    end_omega_1 = np.zeros(3)
    end_accel_1 = np.zeros(3)
    end_alpha_1 = np.zeros(3)

    leader_traj = plan_trajectory(
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
    T_F2L = make_transform_mat(Rx(np.pi), [0, 0, -1 * offset_distance])

    follower_traj = follower_trajectory(leader_traj, T_F2L)
    leader = Astrobee(start_pose_1)
    follower = Astrobee(follower_traj.poses[0])

    kp, kv, kq, kw = 20, 5, 5, 5  # 20, 5, 1, 0.1
    leader_controller = ForceTorqueController(
        leader.id, leader.mass, leader.inertia, kp, kv, kq, kw, dt
    )
    follower_controller = ForceTorqueController(
        follower.id, follower.mass, follower.inertia, kp, kv, kq, kw, dt
    )

    leader_traj.visualize(10)
    follower_traj.visualize(10)

    for i in range(leader_traj.num_timesteps):
        pos_1, orn_1, lin_vel_1, ang_vel_1 = leader.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = follower.dynamics_state
        leader_controller.step(
            pos_1,
            lin_vel_1,
            orn_1,
            ang_vel_1,
            leader_traj.positions[i, :],
            leader_traj.linear_velocities[i, :],
            leader_traj.linear_accels[i, :],
            leader_traj.quaternions[i, :],
            leader_traj.angular_velocities[i, :],
            leader_traj.angular_accels[i, :],
        )
        follower_controller.step(
            pos_2,
            lin_vel_2,
            orn_2,
            ang_vel_2,
            follower_traj.positions[i, :],
            follower_traj.linear_velocities[i, :],
            follower_traj.linear_accels[i, :],
            follower_traj.quaternions[i, :],
            follower_traj.angular_velocities[i, :],
            follower_traj.angular_accels[i, :],
        )
        time.sleep(1 / 120)

    # Stopping mode
    while True:
        pos_1, orn_1, lin_vel_1, ang_vel_1 = leader.dynamics_state
        pos_2, orn_2, lin_vel_2, ang_vel_2 = follower.dynamics_state
        leader_controller.step(
            pos_1,
            lin_vel_1,
            orn_1,
            ang_vel_1,
            leader_traj.positions[-1, :],
            leader_traj.linear_velocities[-1, :],
            leader_traj.linear_accels[-1, :],
            leader_traj.quaternions[-1, :],
            leader_traj.angular_velocities[-1, :],
            leader_traj.angular_accels[-1, :],
        )
        follower_controller.step(
            pos_2,
            lin_vel_2,
            orn_2,
            ang_vel_2,
            follower_traj.positions[-1, :],
            follower_traj.linear_velocities[-1, :],
            follower_traj.linear_accels[-1, :],
            follower_traj.quaternions[-1, :],
            follower_traj.angular_velocities[-1, :],
            follower_traj.angular_accels[-1, :],
        )
        time.sleep(1 / 120)

    input("Press Enter to continue")


# # TODO delete this
# def _test_follower_transform():
#     # pylint: disable=import-outside-toplevel
#     import pybullet
#     from pyastrobee.core.astrobee import Astrobee
#     from pyastrobee.utils.rotations import Rx
#     from pyastrobee.trajectories.planner import plan_trajectory
#     from pyastrobee.control.force_torque_control import ForceTorqueController
#     from pyastrobee.utils.poses import tmat_to_pos_quat
#     from pyastrobee.utils.quaternions import random_quaternion

#     T_F2L = make_transform_mat(Rx(np.pi), [0, 0, -1])

#     pybullet.connect(pybullet.GUI)
#     leader = Astrobee([*np.random.rand(3), *random_quaternion()])
#     T_L2W = pos_quat_to_tmat(leader.pose)
#     T_F2W = T_L2W @ T_F2L
#     follower = Astrobee(tmat_to_pos_quat(T_F2W))
#     input("Enter to continue")


if __name__ == "__main__":
    _test_leader_follower()
    # _test_follower_transform()
